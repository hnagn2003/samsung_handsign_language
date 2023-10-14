import os
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# YouTube video URL
video_url = 'https://www.youtube.com/watch?v=Joi3nnSpbO4'

# Download the YouTube video
yt = YouTube(video_url)
video_stream = yt.streams.filter(file_extension='mp4', resolution='360p').first()
video_title = cleaned_text = yt.title.replace('"', '').replace('?', '').replace(',', '')
download_folder = 'data'

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

video_stream.download(output_path=download_folder)

# Define the time range for the cut
start_time = 0  # Start time in seconds
end_time = 3  # End time in seconds

# Cut the video using moviepy
input_path = os.path.join(download_folder, f'{video_title}.mp4')
output_path = os.path.join(download_folder, f'{video_title}_cut.mp4')

ffmpeg_extract_subclip(input_path, start_time, end_time, targetname=output_path)

print(f'Video downloaded and cut successfully to {output_path}')
