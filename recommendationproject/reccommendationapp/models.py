from django.db import models

class Book(models.Model):
    isbn = models.CharField(max_length=200)
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    year_of_publication = models.IntegerField()
    publisher = models.CharField(max_length=255)
    image_url_s = models.URLField()
    image_url_m = models.URLField()
    image_url_l = models.URLField()

    def __str__(self):
        return self.title

class User(models.Model):
    user_id = models.IntegerField(unique=True)
    location = models.CharField(max_length=255)
    age = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f'User {self.user_id} from {self.location}'

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    book_rating = models.IntegerField()

    def __str__(self):
        return f'User {self.user.user_id} rated {self.book.isbn} with {self.book_rating}'
