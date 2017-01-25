import argparse
import os
import newInterp as interpolator
import movieToPhotos as videoCreator


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-s", "--source", required=True, help="Path to the source image or video")
ap.add_argument("-n", "--next", required=False, help="Specify the second image if using the --picture flag instead of a video")
ap.add_argument("-i", "--imagesize", required=False, help="scale of desired image. Smaller is faster. ex. .5 for half size")
ap.add_argument("-d", "--dir", required=False, help="Output directory. Default './out'")
ap.add_argument('-o', '--output', required=False, help="Output filename if saving is desired. Default out.jpg")

video_or_image = ap.add_mutually_exclusive_group(required=False)
video_or_image.add_argument('--video', dest='video', action='store_true', help="Default video interpolation")
video_or_image.add_argument('--image', dest='video', action='store_false', help="Use for optional image interpolation")

fast_or_slow = ap.add_mutually_exclusive_group(required=False)
fast_or_slow.add_argument('--slow', dest='routine', action='store_true', help="Default interpolation algorithm. Slow.")
fast_or_slow.add_argument('--fast', dest='routine', action='store_false', help="Fast interpolation algorithm.")

ap.set_defaults(routine=True) # true for default slow routine
ap.set_defaults(video=True)
ap.set_defaults(dir="./out")
ap.set_defaults(output="image")
ap.set_defaults(imagesize=1)

args = vars(ap.parse_args())

directory = args['dir']
source = args['source']
filename = args['output']
scalefactor = float(args['imagesize'])

if not os.path.exists(directory):
    os.makedirs(directory)
output = directory + '/' + filename

if args['video'] is False:
    image2 = args['next']
    if image2 is None:
        raise Exception("--next argument required if --image flag is set")

if args['routine'] is True:
    speed = 'slow'
else:
    speed = 'fast'

if args['video'] is False:
    interpolator.execute(speed, source, image2, output, scalefactor)
else:
    videoCreator.videoProcess(speed, source, filename, directory, scalefactor)