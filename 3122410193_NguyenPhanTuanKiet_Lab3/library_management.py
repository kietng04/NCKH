class Author:
    def __init__(self, name, birth_year):
        self.name = name
        self.birth_year = birth_year
    
    def mota(self):
        print(f"Tác giả: {self.name}, Sinh năm: {self.birth_year}")

class Book:
    def __init__(self, title, authors=None):
        self.title = title
        self.authors = authors if authors is not None else []
    
    def mota(self):
        authors_desc = ", ".join([author.name for author in self.authors])
        print(f"Sách: {self.title}, Tác giả: {authors_desc}")

class Borrower:
    def __init__(self, name, id_number, occupation, borrowed_books=None):
        self.name = name
        self.id = id_number
        self.occupation = occupation
        self.borrowed_books = borrowed_books if borrowed_books is not None else []
    
    def mota(self):
        borrowed_titles = ", ".join([book.title for book in self.borrowed_books]) if self.borrowed_books else "Không có"
        print(f"Người mượn: {self.name}, ID: {self.id}, Nghề nghiệp: {self.occupation}, Sách đã mượn: {borrowed_titles}")

class Library:
    def __init__(self, name):
        self.name = name
        self.books = []
        self.authors = []
        self.borrowers = []
    
    def them_sach(self, book):
        self.books.append(book)
    
    def them_tacgia(self, author):
        self.authors.append(author)
    
    def them_nguoimuon(self, borrower):
        self.borrowers.append(borrower)
    
    def mota(self):
        print(f"Thư viện: {self.name}")
        
        print("\nSách:")
        for book in self.books:
            book.mota()
        
        print("\nTác giả:")
        for author in self.authors:
            author.mota()
        
        print("\nNgười mượn:")
        for borrower in self.borrowers:
            borrower.mota()
    
    def dem_sach_kha_dung(self):
        borrowed_books = []
        for borrower in self.borrowers:
            borrowed_books.extend(borrower.borrowed_books)
    
        available_count = 0
        for book in self.books:
            if book not in borrowed_books:
                available_count += 1
                
        return available_count
    
    def sapxep_sach_theo_tieude(self):
        self.books.sort(key=lambda book: book.title)
    
    def tuoi_trungbinh_tacgia(self):
        if not self.authors:
            return 0
        
        current_year = 2025  
        total_age = sum(current_year - author.birth_year for author in self.authors)
        return total_age / len(self.authors)

if __name__ == "__main__":
    author1 = Author("J.K. Rowling", 1965)
    author2 = Author("George R.R. Martin", 1948)
    book1 = Book("Harry Potter", [author1])
    book2 = Book("Game of Thrones", [author2])
    borrower1 = Borrower("Alice", 101, "Student")
    borrower2 = Borrower("Bob", 102, "Teacher")
    borrower1.borrowed_books.append(book1)
    library = Library("Thư viện Trung tâm")
    library.them_sach(book1)
    library.them_sach(book2)
    library.them_tacgia(author1)
    library.them_tacgia(author2)
    library.them_nguoimuon(borrower1)
    library.them_nguoimuon(borrower2)
    library.mota()
    print(f"\nSố sách khả dụng: {library.dem_sach_kha_dung()}")
    library.sapxep_sach_theo_tieude()
    print("\nSách sau khi sắp xếp theo tiêu đề:")
    for book in library.books:
        print(f"- {book.title}")
    print(f"\nTuổi trung bình của tác giả: {library.tuoi_trungbinh_tacgia()}")
