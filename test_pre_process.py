import numpy as np
import cv2, pdb, glob, argparse

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2,masksDL):

	# Convert images to grayscale
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	akaze = cv2.AKAZE_create()
	keypoints1, descriptors1 = akaze.detectAndCompute(im1, None)
	keypoints2, descriptors2 = akaze.detectAndCompute(im2, None)
	
	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
	matches = matcher.match(descriptors1, descriptors2, None)
	
	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False)

	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]
	
	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt
	
	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# Use homography
	height, width, channels = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))
	# copy image in the empty region, unless it is a foreground. Then copy background

	mask_rep=(np.sum(im1Reg.astype('float32'),axis=2)==0)

	im1Reg[mask_rep,0]=im2[mask_rep,0]
	im1Reg[mask_rep,1]=im2[mask_rep,1]
	im1Reg[mask_rep,2]=im2[mask_rep,2]

	mask_rep1=np.logical_and(mask_rep , masksDL[...,0]==255)

	im1Reg[mask_rep1,0]=im1[mask_rep1,0]
	im1Reg[mask_rep1,1]=im1[mask_rep1,1]
	im1Reg[mask_rep1,2]=im1[mask_rep1,2]


	return im1Reg


def adjustExposure(img,back,mask):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	mask = cv2.dilate(mask, kernel, iterations=10)
	mask1 = cv2.dilate(mask, kernel, iterations=300)
	msk=mask1.astype(np.float32)/255-mask.astype(np.float32)/255; msk=msk.astype(np.bool)

	back_tr=back
	back_tr[...,0]=bias_gain(img[...,0],back[...,0],msk)
	back_tr[...,1]=bias_gain(img[...,1],back[...,1],msk)
	back_tr[...,2]=bias_gain(img[...,2],back[...,2],msk)

	return back_tr


def bias_gain(orgR,capR,cap_mask):
	capR=capR.astype('float32')
	orgR=orgR.astype('float32')

	xR=capR[cap_mask]
	yR=orgR[cap_mask]

	gainR=np.nanstd(yR)/np.nanstd(xR);
	biasR=np.nanmean(yR)-gainR*np.nanmean(xR);

	cap_tran=capR*gainR+biasR;

	return cap_tran.astype('float32')


parser = argparse.ArgumentParser(description='Deeplab Segmentation')
parser.add_argument('-i', '--input_dir', type=str, required=True,help='Directory to save the output results. (required)')
args=parser.parse_args()


dir_name=args.input_dir

list_im=glob.glob(dir_name + '/*_img.png'); list_im.sort()


for i in range(0,len(list_im)):

	image = cv2.imread(list_im[i],cv2.IMREAD_COLOR)
	back = cv2.imread(list_im[i].replace('img','back'),cv2.IMREAD_COLOR)
	mask = cv2.imread(list_im[i].replace('img','masksDL'))

	#back_new = adjustExposure(image,back,mask[...,0])

	back_align = alignImages(back, image,mask)

	cv2.imwrite(list_im[i].replace('img','back'),back_align)

str_msg='\nDone: ' + dir_name
print(str_msg)