from pytube import YouTube

def download_video(url, save_path):
    yt = YouTube(url)
    ys = yt.streams.get_highest_resolution()
    ys.download(save_path)

# 예시 URL 및 저장 경로
url = 'https://www.youtube.com/watch?v=your_video_id'
save_path = 'path/to/download'
download_video(url, save_path)
