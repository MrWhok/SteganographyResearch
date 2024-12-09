import random

def generate_random_numbers(n, start=0, end=8):
    random_numbers = [random.randint(start, end) for _ in range(n)]
    return random_numbers

def save_to_txt(numbers, filename):
    with open(filename, 'w') as file:
        for number in numbers:
            file.write(f"{number}\n")

# Example usage:
random_numbers = generate_random_numbers(10000)
save_to_txt(random_numbers, 'random_numbers10.txt')
print("Random numbers saved to random_numbers.txt")