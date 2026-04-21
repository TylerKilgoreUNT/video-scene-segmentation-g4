def format_and_print_results(scene_list):
    """
    Validates and displays the final scene detection results.
    PySceneDetect's scene_list returns tuples of FrameTimecode objects: (start_time, end_time)
    """
    
    num_scenes = len(scene_list)
    scene_word = "scene" if num_scenes == 1 else "scenes"
    
    print("\n" + "="*50)
    print(f"SCENE DETECTION RESULTS ({num_scenes} {scene_word} detected)")
    print("="*50)

    if num_scenes == 0:
        print("No scenes detected.")
    elif num_scenes == 1:
        print("1 continuous scene detected (No cut boundaries found).")
    else:
        print(f"{'Scene #':<10} | {'Start Time':<15} | {'End Time':<15} | {'Frames'}")
        print("-" * 50)
        
        for i, (start_time, end_time) in enumerate(scene_list, 1):
            # FrameTimecode provides .get_timecode() -> HH:MM:SS.mmm
            start_tc = start_time.get_timecode()
            end_tc   = end_time.get_timecode()
            
            # .get_frames() returns the literal 0-indexed frame count
            frames   = f"{start_time.get_frames()} -> {end_time.get_frames()}"
            
            print(f"Scene {i:<4} | {start_tc:<15} | {end_tc:<15} | {frames}")
    
    print("="*50 + "\n")
