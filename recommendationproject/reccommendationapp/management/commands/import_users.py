import csv
import logging
import os
from django.core.management.base import BaseCommand
from django.db import transaction
from reccommendationapp.models import User

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_users_csv(input_file, output_file):
    logger.info(f'Preprocessing CSV: {input_file} -> {output_file}')
    with open(input_file, newline='', encoding='latin1') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter=';')
        writer = csv.writer(outfile, delimiter=';')

        header = next(reader)  # Read the header
        writer.writerow(header)  # Write the header to the output file

        current_row = []
        for line in reader:
            if len(line) == 1 and line[0] and not line[0].isdigit():
                current_row[-1] += '\n' + line[0]
            else:
                if current_row:
                    writer.writerow(current_row)
                current_row = line
        if current_row:
            writer.writerow(current_row)

class Command(BaseCommand):
    help = 'Import user data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The relative path to the users CSV file')

    @transaction.atomic
    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        csv_file_path = os.path.abspath(csv_file)
        logger.info(f'Input CSV file: {csv_file_path}')

        if not os.path.exists(csv_file_path):
            logger.error(f'File not found: {csv_file_path}')
            self.stdout.write(self.style.ERROR(f'File not found: {csv_file_path}'))
            return

        temp_file = os.path.join(os.path.dirname(csv_file_path), 'temp_users.csv')
        logger.info(f'Temporary CSV file: {temp_file}')

        preprocess_users_csv(csv_file_path, temp_file)

        users_to_create = []
        users_to_update = []
        count = 0
        batch_size = 10000  # Larger batch size for efficiency

        # Cache existing users
        existing_users = {user.user_id: user for user in User.objects.all()}

        with open(temp_file, newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            header = next(reader)  # Skip the header row

            for row in reader:
                try:
                    user_id, location, age = row
                    user_id = int(user_id)
                    age = int(age) if age.isdigit() else None

                    if user_id in existing_users:
                        # Update existing user
                        user = existing_users[user_id]
                        user.location = location.replace('\n', ' ')
                        user.age = age
                        users_to_update.append(user)
                    else:
                        # Create new user
                        users_to_create.append(
                            User(
                                user_id=user_id,
                                location=location.replace('\n', ' '),
                                age=age
                            )
                        )
                    count += 1

                    # Bulk create or update users in batches
                    if len(users_to_create) >= batch_size:
                        User.objects.bulk_create(users_to_create)
                        logger.info(f'{len(users_to_create)} user records created...')
                        users_to_create = []

                    if len(users_to_update) >= batch_size:
                        User.objects.bulk_update(users_to_update, ['location', 'age'])
                        logger.info(f'{len(users_to_update)} user records updated...')
                        users_to_update = []

                except ValueError as e:
                    logger.error(f'Error processing row {row}: {e}')
                    continue

            # Create or update remaining users
            if users_to_create:
                User.objects.bulk_create(users_to_create)
                logger.info(f'{len(users_to_create)} user records created...')

            if users_to_update:
                User.objects.bulk_update(users_to_update, ['location', 'age'])
                logger.info(f'{len(users_to_update)} user records updated...')

        self.stdout.write(self.style.SUCCESS(f'Total {count} user records processed successfully'))
