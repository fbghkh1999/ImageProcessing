import cv2
import numpy as np
import matplotlib.pyplot as plt




feature_extractor = 'surf'
feature_matching = 'knn'


def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if  method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    
    return bf



def detectAndDescribe(image, method=None):
   
    
    assert method is not None, "You need to define a feature detection method. Values are: 'surf'"
    
    if method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
        
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)


def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
   
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    
    for m,n in rawMatches:
     
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
  
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

       
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None

trainImg= cv2.imread('/home/f/Desktop/term6/binaii/projectnahaii/panaroma-stitching/yard3.jpg')
queryImg= cv2.imread('/home/f/Desktop/term6/binaii/projectnahaii/panaroma-stitching/yard-05.jpg')
trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)
kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor )
kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor )
if feature_matching == 'knn':
    matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
    img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,np.random.choice(matches,100),
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
(matches, H, status) = M
width = trainImg.shape[1] + queryImg.shape[1]
height = trainImg.shape[0] + queryImg.shape[0]

h1, w1 = queryImg.shape[:2]
h2, w2 = trainImg.shape[:2]
pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
pts2_ = cv2.perspectiveTransform(pts2, H)
pts = np.concatenate((pts1, pts2_), axis=0)
[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
t = [-xmin, -ymin]
Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

result = cv2.warpPerspective(trainImg, Ht.dot(H), (xmax - xmin, ymax - ymin))
result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = queryImg

A= np.nonzero(result)
yy = A[0]
xx = A[1]
minX = np.min(xx)
maxX = np.max(xx)
minY = np.min(yy)
maxY = np.max(yy)

result = result[minY:maxY, minX:maxX]

cv2.imwrite("/home/f/Desktop/term6/binaii/projectnahaii/panaroma-stitching/yard3.jpg", result)
