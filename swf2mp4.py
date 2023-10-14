import os
import ffmpeg

# Specify the directory containing the SWF files
swf_directory = '/home/hngan/Desktop/Project/samsung/WLASL/start_kit/raw_videos'

# Specify the directory where you want to save the MP4 files
mp4_directory = '/home/hngan/Desktop/Project/samsung/WLASL/start_kit/raw_videos'

# List all SWF files in the specified directory
swf_files = [f for f in os.listdir(swf_directory) if f.endswith('.swf')]

# Loop through each SWF file and convert to MP4
for swf_file in swf_files:
    input_path = os.path.join(swf_directory, swf_file)
    output_file = os.path.splitext(swf_file)[0] + '.mp4'
    output_path = os.path.join(mp4_directory, output_file)

    try:
        # Use ffmpeg-python to perform the conversion
        ffmpeg.input(input_path).output(output_path).run()
        print(f"Converted {swf_file} to {output_file}")
    except ffmpeg.Error as e:
        print(f"Error converting {swf_file}: {e.stderr}")
