import os
import re
import base64
import time
import mimetypes
import logging
import requests

from src.settings.config import settings

logger = logging.getLogger(__name__)

IMAGE_PATTERN = re.compile(r'[a-fA-F0-9]+_img\.\w+', re.IGNORECASE)


def submit_ocr(pdf_path: str) -> str:
    """Submit a PDF to the Datalab Marker OCR API. Returns the polling check_url."""
    mime_type, _ = mimetypes.guess_type(pdf_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    headers = {"X-Api-Key": settings.CHANDRA_OCR_API_KEY}

    with open(pdf_path, "rb") as f:
        response = requests.post(
            settings.CHANDRA_OCR_API_URL,
            headers=headers,
            files={"file": (os.path.basename(pdf_path), f, mime_type)},
            data={"output_format": "markdown"},
            timeout=60,
        )

    if response.status_code != 200:
        raise RuntimeError(f"OCR upload failed: {response.text}")

    data = response.json()
    check_url = data.get("request_check_url")
    if not check_url:
        raise RuntimeError(f"No check URL in OCR response: {data}")

    logger.info("OCR job submitted. Polling: %s", check_url)
    return check_url


def poll_ocr(check_url: str) -> dict:
    """Poll the OCR job until complete. Returns the full result dict."""
    headers = {"X-Api-Key": settings.CHANDRA_OCR_API_KEY}
    polls = 0

    while polls < settings.MAX_POLL:
        poll = requests.get(check_url, headers=headers, timeout=30).json()
        if poll["status"] == "complete":
            return poll
        elif poll["status"] == "failed":
            raise RuntimeError("OCR job failed")
        logger.info("OCR processing... (poll %d)", polls + 1)
        time.sleep(settings.POLL_INTERVAL)
        polls += 1

    raise RuntimeError(f"OCR polling timed out after {settings.MAX_POLL} attempts")


def save_ocr_images(ocr_result: dict, ocr_text: str) -> dict[str, str]:
    """Decode base64 images from OCR result, save to disk, return {name: path} dict."""
    images_dir = settings.EXTRACTED_IMAGES_DIR
    os.makedirs(images_dir, exist_ok=True)

    images_dict = ocr_result.get("images", {})
    saved_images = {}

    for img_name, img_b64 in images_dict.items():
        raw_b64 = img_b64.split(",")[1] if "," in img_b64 else img_b64
        img_path = os.path.join(images_dir, img_name)
        with open(img_path, "wb") as img_f:
            img_f.write(base64.b64decode(raw_b64))
        saved_images[img_name] = img_path

    # Catch image refs in markdown not in the images dict
    for match in IMAGE_PATTERN.findall(ocr_text):
        if match not in saved_images:
            dest = os.path.join(images_dir, match)
            if not os.path.exists(dest):
                open(dest, "wb").close()
            saved_images[match] = dest

    return saved_images


def extract_text_and_images(pdf_path: str) -> tuple[str, dict[str, str]]:
    """Full OCR pipeline: submit -> poll -> save images. Returns (ocr_text, saved_images)."""
    check_url = submit_ocr(pdf_path)
    ocr_result = poll_ocr(check_url)
    ocr_text = ocr_result.get("markdown", "")
    saved_images = save_ocr_images(ocr_result, ocr_text)
    logger.info("OCR complete: %d chars, %d images", len(ocr_text), len(saved_images))
    return ocr_text, saved_images