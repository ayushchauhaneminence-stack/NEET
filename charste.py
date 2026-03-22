def count_characters(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return len(content)
    except FileNotFoundError:
        print("File not found!")
        return 0
    except Exception as e:
        print("Error:", e)
        return 0


if __name__ == "__main__":
    file_path = "Full_Chapter.txt"
    char_count = count_characters(file_path)
    print(f"Total characters in '{file_path}': {char_count}")