import os
import subprocess
import logging
from django.core.management.base import BaseCommand, CommandError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run the full import and pre-processing sequence'

    def handle(self, *args, **options):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        commands_and_scripts = [
            ['python', 'manage.py', 'import_users', os.path.join(base_dir, 'data', 'BX-Users.csv')],
            ['python', os.path.join(base_dir, 'users-pre-processing.py')],
            ['python', 'manage.py', 'import_books', os.path.join(base_dir, 'data', 'BX-Books.csv')],
            ['python', 'manage.py', 'import_ratings', os.path.join(base_dir, 'data', 'BX-Book-Ratings.csv')],
            ['python', os.path.join(base_dir, 'users-pre-processing.py')],
            ['python', os.path.join(base_dir, 'books-pre-processing.py')],
            ['python', os.path.join(base_dir, 'ratings-pre-processing.py')],

        ]

        for command in commands_and_scripts:
            try:
                logger.info(f"Running command: {' '.join(command)}")
                subprocess.check_call(command)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with error: {e}")
                raise CommandError(f"Command failed: {' '.join(command)}")

        self.stdout.write(self.style.SUCCESS('Successfully ran all commands and scripts'))

if __name__ == "__main__":
    Command().handle()
