import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def load_model(model_path: str):
    '''Load the YOLO model.'''

    return YOLO(model_path)


def detect_objects(model, image_path: str):
    '''Detect objects in the image using the YOLO model.'''

    return model(image_path)


def display_results(result):
    '''Display detailed prediction insights.'''

    box_all = result.boxes
    for i, box in enumerate(box_all):
        label = int(box.cls.numpy()[0])
        confidence = float(box.conf.numpy()[0])
        coordinates = box.xyxyn.numpy()[0]

        # Calculate bounding box dimensions and area
        x_min, y_min, x_max, y_max = coordinates
        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        print(f'Object {i + 1}:')
        print(
            f'  Type: {label:2d}, '
            f'Confidence: {confidence:.2f}'
        )
        print(
            f'  Coordinates: '
            f'x_min={x_min:.3f}, '
            f'y_min={y_min:.3f}, '
            f'x_max={x_max:.3f}, '
            f'y_max={y_max:.3f}'
        )
        print(
            f'  Dimensions: '
            f'width={width:.3f}, '
            f'height={height:.3f}, '
            f'Area={area:.3f}'
        )
        print()


def save_result(result, output_path: str):
    '''Save the result image.'''

    result.save(output_path)
    print(f'Saved predicted result to {output_path}')


def show_result_with_matplotlib(result):
    '''Display the result image with bounding boxes using matplotlib.'''

    # Get the image and boxes
    img = result.plot()  # Plot the image with bounding boxes
    img = np.array(img)  # Convert to numpy array for matplotlib

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Object Detection Result')
    plt.show()


def detect_and_display(model_path: str, image_paths: list):
    '''Detect objects in multiple images using a YOLO model and display the results.'''

    # Input and output file paths
    # example_model_path = 'models/bottle/Run7_yolo11n_512_SGD/best.pt'
    # example_image_name = 'tests/bottle_input.png'
    # example_result_name = 'tests/bottle_output_Run7_yolo11n_512_SGD.jpg'

    # Load the model
    model = load_model(model_path)

    # Detect objects
    os.makedirs('output', exist_ok=True)
    for image_path in image_paths:
        print(f'\nProcessing image: {image_path}')
        result_all = detect_objects(model, image_path)

        # Display detailed results
        print('Result:')
        result = result_all[0]
        display_results(result)

        # Save the result
        save_result(result, f'output/{os.path.basename(image_path)}')

        # Show the result with matplotlib
        show_result_with_matplotlib(result)


if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='YOLO Object Detection and Result Visualization')
    parser.add_argument('model_path', type=str,
                        help='Path to the YOLO model file')
    parser.add_argument('image_paths', nargs='+', help='Paths to input images')
    args = parser.parse_args()

    # Call the detection and display function
    detect_and_display(args.model_path, args.image_paths)
