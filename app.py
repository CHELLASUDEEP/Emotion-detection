from flask import Flask, render_template, request, redirect, url_for, send_file
import subprocess
import os

app = Flask(__name__)


# Function to perform inference on uploaded video
def perform_inference(video_path):
    output_dir = 'output'  # Output directory name
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    output_filename = os.path.basename(video_path.split('.')[0]) + "_output.mp4"  # Output video filename
    output_path = os.path.join(output_dir, output_filename)  # Output video path

    # Run inference command and save the output to the specified path
    command = f"python detect.py --source {video_path} --weights best.pt --imgsz 640 --project output/ --name {video_path.split('/')[1]} "
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    return output_path  # Return the path of the output video


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser should also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file
            file_path = 'uploads/' + file.filename
            file.save(file_path)

            # Perform inference on the uploaded file
            inference_result = perform_inference(file_path)

            return inference_result


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
