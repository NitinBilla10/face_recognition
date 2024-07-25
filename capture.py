import cv2
import os

# Create directory to save photos
if not os.path.exists('photos'):
    os.makedirs('photos')


cap = cv2.VideoCapture(0)

num_people = 1 
num_images_per_person = 20

for person_id in range(num_people):
    person_folder = os.path.join('photos', f'person_{person_id}')
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    
    count = 0
    while count < num_images_per_person:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow('Capture Photo', frame)
        
        # Save photo
        photo_path = os.path.join(person_folder, f'photo_{count}.jpg')
        cv2.imwrite(photo_path, frame)
        print(f'Captured {photo_path}')
        
        count += 1
        
        # Wait for 1 second
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
