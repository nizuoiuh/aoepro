import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab
import pytesseract

# Specify the path to the Tesseract executable if not in system PATH
# pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'

# Function to capture the game screen
def capture_screen():
    screen = ImageGrab.grab(bbox=(0, 0, screen_resolution[0], screen_resolution[1]))
    return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

# Function to extract text from an image
def extract_text(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)

    # Threshold the image to get only white regions
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_text = cv2.bitwise_and(image, image, mask=white_mask)

    # Convert the result to grayscale
    gray = cv2.cvtColor(white_text, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)

    text = pytesseract.image_to_string(threshold)
    return text.strip(), threshold  # Return both the text and the thresholded image


# Function to draw rectangles on the screen
def draw_rectangles(image, regions):
    for region in regions:
        cv2.rectangle(image, (region[0], region[1]), (region[2], region[3]), (0, 255, 0), 2)

# Set your screen resolution
screen_resolution = (1920, 1080)  # Change this to your screen resolution

# Define coordinates for the villager queue and game time
villager_queue_coordinates = (0, 80, 300, 120)  # Adjust these coordinates based on your screen
game_time_coordinates = (1130, 0, 1220, 40)  # Adjust these coordinates based on your screen

# Main loop
while True:
    # Capture the game screen
    game_screen = capture_screen()

    # Draw rectangles around regions of interest
    draw_rectangles(game_screen, [villager_queue_coordinates, game_time_coordinates])

    # Extract text from the regions
    villager_queue_text, villager_queue_threshold = extract_text(game_screen[villager_queue_coordinates[1]:villager_queue_coordinates[3],
                                                                            villager_queue_coordinates[0]:villager_queue_coordinates[2]])

    game_time_text, game_time_threshold = extract_text(game_screen[game_time_coordinates[1]:game_time_coordinates[3],
                                                                        game_time_coordinates[0]:game_time_coordinates[2]])

    # Print the extracted information
    print("Villager Queue:", villager_queue_text)
    print("Game Time:", game_time_text)

    # Display the screen with rectangles
    cv2.imshow('Game Screen', game_screen)

    # Display the thresholded images of detected text
    cv2.imshow('Villager Queue Text', villager_queue_threshold)
    cv2.imshow('Game Time Text', game_time_threshold)

    # Wait for a short period before capturing the next frame
    if cv2.waitKey(5000) & 0xFF == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()
