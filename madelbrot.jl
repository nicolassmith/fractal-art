using Images, Colors, ColorSchemes
using Pipe
using CuArrays, GPUArrays
using ImageFiltering

# Define the function for counting the steps to divergence.

function recurse_steps_to_satisfy(
    f::Function, condition::Function, max_steps::Integer)
    z = Complex(0, 0)
    for i = 1:max_steps
        z = f(z)
        if condition(z)
            return i
        end
    end
    return max_steps + 1
end

function mandelbrot_function(c::Complex)
    return (z) -> z*z + c
end

function mandelbrot_condition(z::Complex)
    return abs2(z) >= 4
end

function mandelbrot_steps(c::Complex, max_steps::Integer)
    return recurse_steps_to_satisfy(
        mandelbrot_function(c), mandelbrot_condition, max_steps)
end

function get_cmap(colorscheme::ColorScheme, max_steps)
    return [colorscheme[num_steps / (max_steps + 1)] 
        for num_steps in 1:max_steps+1]
end

# Color space.
colorscheme = ColorSchemes.inferno
max_steps = 5000

# Lay out the complex domain to evaluate.
pixel_size = 0.0000001

resample_factor = 2
blur_kernel = Kernel.gaussian(resample_factor/2)

width = 1920
height = 1080

center = (-0.75393, 0.05)

horizontal_extent = width * pixel_size
vertical_extent = height * pixel_size
real_domain = center[1]-horizontal_extent/2:pixel_size/resample_factor:center[1]+horizontal_extent/2
imaginary_domain = center[2]-vertical_extent/2:pixel_size/resample_factor:center[2]+vertical_extent/2
# The imaginary domain is reversed so that the image comes out with the correct orientation.
imaginary_domain = reverse(imaginary_domain)

complex_domain = [Complex(x, y) for y in imaginary_domain, x in real_domain]
color_map = get_cmap(colorscheme, max_steps)

@time begin
    # Casting as CuArray sends the computation to the GPU.
    steps = @pipe complex_domain |> CuArray |> mandelbrot_steps.(_, max_steps) |> Array
    image = color_map[steps]
    image = imfilter(image, blur_kernel)
    image = image[1:resample_factor:end-1,1:resample_factor:end-1]
end

save("image.png", image)

image