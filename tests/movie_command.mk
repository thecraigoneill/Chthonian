ffmpeg -framerate 10  -pattern_type glob -i "impact23_*.png" -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4
	
