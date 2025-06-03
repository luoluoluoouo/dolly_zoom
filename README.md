# Dolly Zoom
113-2 Machine vision final project

## Usage
> dolly_zoom.py

```
main(
    left_path= "path/to/your/left_image",
    right_path= "path/to/your/right_image",
    output_gif_path= "result/dolly_zoom.gif",
    fg_strength_range= (0.0, 0.2),
    bg_strength_range= (0.0, 0.3),
    steps= 15,
    num_disparities= 16 * 11,
    block_size= 11,
    disparity_threshold= 35,
    invert_mask=False,
    kernel=np.ones((5, 5), np.uint8)
)
```