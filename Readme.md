## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/moaaz12-web/Cattle-detection-.git
    cd cattle-image-recognition
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download weights for Alexnet:

    ```
    Before running the application, download the AlexNet model weights from the drive link and place it in the working directory
    https://drive.google.com/file/d/1PnKTyFy9yBtaossFSwrRGsb5S6CPVhEK/view?usp=sharing
    ```

4. Run the application:

    ```bash
    python tk.py
    ```

5. Upload an image through the interface, and witness the advanced image processing workflow:
   - The user-uploaded image undergoes a series of processing functions.
   - The processed image is sent to the AlexNet model, providing a string output indicating the presence of a cow in the image.
   - Subsequently, the image is forwarded to the YOLOv8 model, specifically trained on a cattle dataset.
   - YOLOv8 counts the number of cattle in the image and overlays bounding boxes to visually indicate their locations.
   - The entire process is seamlessly displayed on the Tkinter interface, offering a comprehensive and interactive user experience.

