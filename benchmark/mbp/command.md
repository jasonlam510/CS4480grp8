```Bash
Usage: ./extract_frames.sh -i <input_folder> -o <output_folder> [options]

Required arguments:
  -i, --input_path    Input folder containing MP4 videos
  -o, --output_path   Output folder for extracted frames

Optional arguments:
  -r, --resize_factor Resize factor (e.g., 2.0 to halve resolution, 0 for no resize) [default: 0]
  -f, --fps           Frames per second to extract [default: 1]
  -t, --threads       Number of threads for FFmpeg [default: 0]
  -p, --parallel      Number of videos to process in parallel [default: 1]
  --ffmpeg_path       Custom FFmpeg path [default: system ffmpeg]
  --benchmark         Run benchmark mode: test multiple parallel job counts [default: false]
  -h, --help          Show this help message

Example:
  ./extract_frames.sh -i /videos -o /frames -r 2.0 -f 1 -t 2 -p 4
```

```Bash
./extract_frames.sh -i /Users/jason/Github/CS4480grp8/data/TheArcticSkies -o /Users/jason/Github/CS4480grp8/data/TheArcticSkies_frames -r 10 -f 1 --benchmark  
```