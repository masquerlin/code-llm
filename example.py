def edit_book(library):
    """
    编辑图书信息的函数。

    """
    isbn = input("Enter ISBN of the book to edit: ")
    book = library.find_book_by_isbn(isbn)
    if book:
        print("Book found. Enter new details:")
        title = input("Enter new title (press Enter to keep existing): ")
        author = input("Enter new author (press Enter to keep existing): ")
        if title:
            book.title = title
        if author:
            book.author = author
        print("Book details updated successfully!")
    else:
        print("Book not found.")

def count_books(library):
    """
    统计图书馆中图书总数的函数。

    """
    print(f"Total books in the library: {len(library.books)}")

def clear_library(library):
    """
    清空图书馆中所有图书的函数。

    """
    library.books = []
    print("All books removed from the library.")

def save_library(library):
    """
    将图书馆数据保存到文件的函数。

    """
    with open("library.txt", "w") as file:
        for book in library.books:
            file.write(f"{book.title},{book.author},{book.isbn}\n")
    print("Library data saved to file.")

def load_library(library):
    """
    从文件中加载图书馆数据的函数。
    """
    try:
        with open("library.txt", "r") as file:
            for line in file:
                title, author, isbn = line.strip().split(",")
                book = Book(title, author, isbn)
                library.add_book(book)
        print("Library data loaded from file.")
    except FileNotFoundError:
        print("Library data file not found.")

def check_book_availability(library):
    """
    检查图书是否在图书馆中的函数。
    """
    isbn = input("Enter ISBN of the book to check availability: ")
    book = library.find_book_by_isbn(isbn)
    if book:
        print("Book is available in the library.")
    else:
        print("Book is not available in the library.")

def borrow_book(library):
    """
    借阅图书的函数。
    """
    isbn = input("Enter ISBN of the book to borrow: ")
    if library.remove_book(isbn):
        print("Book borrowed successfully!")
    else:
        print("Book not found or already borrowed.")

def return_book(library):
    """
    归还图书的函数。
    """
    isbn = input("Enter ISBN of the book to return: ")
    book = Book("Sample", "Sample Author", isbn)
    library.add_book(book)
    print("Book returned successfully!")

def list_books_by_title(library):
    """
    根据标题列出图书的函数。
    """
    title = input("Enter book title to search: ")
    books = library.find_books_by_title(title)
    if books:
        print("Books found with the title:")
        for book in books:
            print(f"Title: {book.title}, Author: {book.author}, ISBN: {book.isbn}")
    else:
        print("No books found with the title.")

def list_books_by_isbn(library):
    """
    根据ISBN列出图书的函数。
    """
    isbn = input("Enter ISBN to search book: ")
    book = library.find_book_by_isbn(isbn)
    if book:
        print(f"Book found - Title: {book.title}, Author: {book.author}, ISBN: {book.isbn}")
    else:
        print("Book not found.")

def list_books_by_author(library):
    """
    根据作者列出图书的函数。

    """
    author = input("Enter author name to search books: ")
    books = library.find_books_by_author(author)
    if books:
        print("Books found for the author:")
        for book in books:
            print(f"Title: {book.title}, Author: {book.author}, ISBN: {book.isbn}")
    else:
        print("No books found for the author.")

def main():
    """
    主函数，控制图书管理系统的整体流程。
    """
    library = Library()
    load_library(library)  # Load existing library data if available
    while True:
        print("\n1. Add book to library")
        print("2. Remove book from library")
        print("3. Find book by title")
        print("4. Find books by author")
        print("5. Edit book details")
        print("6. Display all books")
        print("7. Count total books")
        print("8. Clear library")
        print("9. Save library to file")
        print("10. Check book availability")
        print("11. Borrow a book")
        print("12. Return a book")
        print("13. List books by title")
        print("14. List books by ISBN")
        print("15. List books by author")
        print("16. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            add_book_to_library(library)
        elif choice == "2":
            remove_book_from_library(library)
        elif choice == "3":
            find_book_by_title(library)
        elif choice == "4":
            find_books_by_author(library)
        elif choice == "5":
            edit_book(library)
        elif choice == "6":
            display_all_books(library)
        elif choice == "7":
            count_books(library)
        elif choice == "8":
            clear_library(library)
        elif choice == "9":
            save_library(library)
        elif choice == "10":
            check_book_availability(library)
        elif choice == "11":
            borrow_book(library)
        elif choice == "12":
            return_book(library)
        elif choice == "13":
            list_books_by_title(library)
        elif choice == "14":
            list_books_by_isbn(library)
