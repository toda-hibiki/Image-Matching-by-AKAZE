import cv2  #OpenCVのインポート

img1 = cv2.imread('image1.jpg') #img1の読み出し
img2 = cv2.imread('image2.jpg') #img2の読み出し

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #img1をグレースケールで読み出し
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #img2をグレースケールで読み出し

akaze = cv2.AKAZE_create() #AKAZE検出器の生成
kp1, des1 = akaze.detectAndCompute(gray1, None) #gray1にAKAZEを適用、特徴点を検出
kp2, des2 = akaze.detectAndCompute(gray2, None) #gray2にAKAZEを適用、特徴点を検出

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#Match descriptorsを生成
matches = bf.match(des1, des2)
#BFMatcherオブジェクトの生成
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#matchesをdescriptorsのdistance順(似ている順)にsortする 
matches = sorted(matches, key = lambda x:x.distance)
#img3に検出結果(最初の10点)を描画
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('match.png',img3)
