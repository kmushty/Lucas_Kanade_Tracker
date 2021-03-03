import glob
import numpy as np
import cv2


class CoordinateSave:
    def _init_(self):
        self.pts = []

    def pic_select(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            Dx, Dy = x, y
            self.pts = [(Dx, Dy)]

        elif event == cv2.EVENT_LBUTTONUP:
            Ux, Uy = x, y
            self.pts.append((Ux, Uy))
            cv2.rectangle(first_frame, self.pts[0], self.pts[1], (0, 255, 0), 2)
            cv2.imshow("Select Object to Track", first_frame)


CdSave = CoordinateSave()


def object_box_click(first_frame):
    img = first_frame.copy()

    cv2.namedWindow("Select Object to Track")
    cv2.setMouseCallback("Select Object to Track", CdSave.pic_select)
    cv2.imshow("Select Object to Track", img)
    cv2.waitKey(0)
    object_select = CdSave.pts
    print(object_select)
    cv2.destroyAllWindows()
    return object_select


def create_tmp(frame, rect):
    ROI = frame[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    return ROI


def convert_lab(image):
    clahe = cv2.createCLAHE(clipLimit=1., tileGridSize=(1,1))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    image2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image2


def warp_img(img, w, rect):
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]

    warp_img = np.zeros((y2 - y1, x2 - x1), dtype=np.float64)
    new = np.vstack([w, [0, 0, 1]])
    for indy, y in enumerate(range(y1, y2)):
        for indx, x in enumerate(range(x1, x2)):
            warp = np.matmul(new, [x, y, 1])
            warpX = warp[0] / warp[2]
            warpY = warp[1] / warp[2]
            warp_img[indy, indx] = img[int(round(warpY)), int(round(warpX))]
    return warp_img


def tracker(img, tmp, rect, p, count):

    new_img = img = cv2.GaussianBlur(img, (3, 3), 0)
    threshold = 1
    iterations = 0

    while threshold > 0.05:
        w = np.array([[1 + p[0, 0], p[2, 0], p[4, 0]], [p[1, 0], 1 + p[3, 0], p[5, 0]]])
        warped_img = warp_img(np.float64(new_img), w, rect)

        grad_x = cv2.Sobel(np.float64(img), cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(np.float64(img), cv2.CV_64F, 0, 1, ksize=5)

        Ix = warp_img(grad_x, w, rect)
        Iy = warp_img(grad_y, w, rect)

        # Error calculation
        Err = tmp.astype(np.float64) - warped_img.astype(np.float64)
        errorMean = np.mean(Err)
        for row in range(Err.shape[0]):
            for col in range(Err.shape[1]):
                if abs(Err[row, col]) < errorMean:
                    Err[row, col] = Err[row, col] * 0.5
                else:
                    Err[row, col] = np.sign(Err[row, col]) * errorMean

        h, w = warped_img.shape
        T = np.vstack((Ix.ravel(), Iy.ravel())).T
        sd = np.zeros([h * w, 6])
        hessian = np.zeros((6, 6))
        sd_err = np.zeros((6, 1))
        for i in range(h):
            for j in range(w):
                I_indiv = np.array([T[i * w + j]]).reshape(1, 2)

                jac_indiv = np.array([[j, 0, i, 0, 1, 0],
                                      [0, j, 0, i, 0, 1]])
                sd[i * w + j] = np.matmul(I_indiv, jac_indiv)

        for i in range(h):
            for j in range(w):
                sd_T = np.transpose(sd[i])
                sd_err = sd_err + (sd_T * Err[i][j])
                hessian = hessian + np.matmul(sd.T, sd)

        H_inv = np.linalg.inv(hessian)

        # Calculating the delta_p
        dp = np.multiply(H_inv, sd_err)

        # Robustness
        if iterations > 3 and count > 3:
            gamma = np.diag([0.5, 0.5, 0.1, 0.1, -0.0005, 0.0005])

        else:
            gamma = np.diag([0.5, 0.5, 0.1, 0.1, -0.001, 0.001])

        p = p + np.dot(gamma, dp)
        threshold = np.linalg.norm(dp)
        iterations += 1

    return p


path = "Data_Sets/DragonBaby/img/*.jpg"
frames_list = [img for img in glob.glob(path)]
frames_list.sort()
first_frame = cv2.imread(frames_list[0])
first_frame = convert_lab(first_frame)
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
temp_mean = np.mean(first_frame)

# Finding the template
rect_tup = object_box_click(first_frame)
rect = [rect_tup[0][0], rect_tup[0][1], rect_tup[1][0], rect_tup[1][1]]
Tmp_img = create_tmp(first_frame, rect_tup)
p = np.zeros((6, 1))

detected_frames = []
count = 0

for img in frames_list:
    image = cv2.imread(img)
    image = convert_lab(image)
    cv2.waitKey(1)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_mean = np.mean(gray_img)
    current_img = (gray_img * (temp_mean / img_mean)).astype(float)
    count += 1

    p = tracker(current_img, Tmp_img, rect, p, count)

    p1, p2, p3, p4, p5, p6 = p[0,0], p[1,0], p[2,0], p[3,0], p[4,0], p[5,0]


    W = np.array([[1 + p1, p3, p5], [p2, 1 + p4, p6]])

    new_mat_1 = np.array([[rect_tup[0][0]], [rect_tup[0][1]], [1]])
    new_mat_2 = np.array([[rect_tup[1][0]], [rect_tup[1][1]], [1]])

    wx1 = np.matmul(W, new_mat_1).astype(int)
    wx2 = np.matmul(W, new_mat_2).astype(int)

    cv2.rectangle(image, (wx1[0], wx1[1]), (wx2[0], wx2[1]), color=(0,255,0))
    detected_frames.append(image)

    cv2.imshow('Tracked', image)
    cv2.waitKey(0)

#Generating the video
#source = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output_car.avi', source, 5.0, (720, 480))
#for iter1 in detected_frames:
#    out.write(iter1)
#    cv2.waitKey(15)
#    out.release()