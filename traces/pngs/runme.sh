#ubuntu
avconv -framerate 10 -f image2 -i multirobot-path-%d.png -c:v h264 -vf "scale=718:-1" -crf 1 out.mov
#windows
ffmpeg -framerate 290/15 -i multirobot-path-%d.png -c:v libx264 -vf "scale=718:
-1" -r 30 -pix_fmt yuv420p out.mp4