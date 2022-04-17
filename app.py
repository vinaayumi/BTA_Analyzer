#Usage: python app.py
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
from matplotlib import pyplot as plt
import sys
from  PIL  import Image, ImageEnhance

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    # Image directory
    directory = r'C:\Users\vinaa\Python Project\Hibah 2021\IDA-Poltekkes\BTA Analyzer Web-based2\uploads'
    os.chdir(directory)
    print(os.listdir(directory))
    global cv2
    oriimage = cv2.imread(file)
    
    crop1 = oriimage[150:3080, 800:3770]
    img = cv2.resize(crop1, (0, 0), fx = 0.25, fy = 0.25)
    cv2.imwrite('img_crop.png',img)
    original = img.copy()

    im = Image.open('img_crop.png')
    im_out = ImageEnhance.Color(im).enhance(1.5)
    im_out.save('im_enhance.jpg')
    img = cv2.imread('im_enhance.jpg')

    # output image with contours
    image_contours = img.copy()
    counter = 0

    #l = int(max(5, 6))
    #u = int(min(6, 6))

    ed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv.GaussianBlur(img, (21, 51), 3)
    #edges = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
    #edges = cv.Canny(edges, l, u)

    #_, thresh = cv.threshold(edges, 0, 255, cv.THRESH_BINARY  + cv.THRESH_OTSU)
    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))    
    #mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)

    ret, thresh1 = cv2.threshold(ed, 150, 255, cv2.THRESH_BINARY)
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours2[0] if len(contours2) == 2 else contours2[1]
    cv2.drawContours(thresh1, [cnt], -1, (255, 255, 255), 2, cv2.LINE_AA)

    mask = thresh1
    #cv.imshow('mask', mask)
    #cv.waitKey()

    data = mask.tolist()
    #sys.setrecursionlimit(10**8)
    for i in  range(len(data)):
        for j in  range(len(data[i])):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
        for j in  range(len(data[i])-1, -1, -1):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
    image = np.array(data)
    image[image !=  -1] =  255
    image[image ==  -1] =  0

    mask = np.array(image, np.uint8)

    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask ==  0] =  255
    cv2.imwrite('bg.png', result)

    img = Image.open('bg.png')
    img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] ==  255  and item[1] ==  255  and item[2] ==  255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save("img.png", "PNG")

    image = cv2.imread('img.png')

    def zoom(img, zoom_factor=2):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
        
    zoomed_and_cropped = zoom(image, 2)
    #cv2.imshow('zoomed_and_cropped', zoomed_and_cropped)
    #cv2.waitKey()

    image_edged = cv2.erode(image, None, iterations=1)
    image_edged = cv2.dilate(image_edged, None, iterations=1)
    image_edged = cv2.erode(image_edged, None, iterations=1)
    image_edged = cv2.dilate(image_edged, None, iterations=1)
    image_edged = cv2.erode(image_edged, None, iterations=1)
    image_edged = cv2.dilate(image_edged, None, iterations=1)
    image_edged = cv2.erode(image_edged, None, iterations=1)
    cv2.imwrite('img_erode.png',image_edged)

    filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # Applying cv2.filter2D function on our Cybertruck image
    image_edged=cv2.filter2D(image_edged,-1,filter)
    #cv2.imshow("sharpen",image_edged)
    #cv2.waitKey(0)
    cv2.imwrite('img_sharpen.png',image_edged)

    hsv = cv2.cvtColor(image_edged, cv2.COLOR_BGR2HSV)
    cv2.imwrite('img_hsv.png',hsv)

    '''lower = np.array([114,50,57], dtype=np.uint8)
    upper = np.array([175,150,190], dtype=np.uint8)'''

    lower = np.array([118,117,52], dtype=np.uint8)
    upper = np.array([143,168,202], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    #cv2.imshow("mask1", mask) 
    #cv2.waitKey(0)
    cv2.imwrite('color_mask.png',mask)

    #mouse callback function
    def pick_color(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = image_src[y,x]

            #you might want to adjust the ranges(+-10, etc):
            upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
            lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
            print(pixel, lower, upper)

            image_mask = cv2.inRange(image_src,lower,upper)
            #cv2.imshow("mask_rgb",image_mask)

    import sys
    import imutils
    global image_hsv, pixel # so we can use it in mouse callback
    image_src = imutils.resize(image_edged, height=800)
    if image_src is None:
        print ("the image read is None............")

    ## NEW ##
    cv2.namedWindow('bgr')
    cv2.setMouseCallback('bgr', pick_color)

    # now click into the hsv img , and look at values:
    #cv2.imshow("bgr",image_src)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #[112 120 144] [102 110 104] [122 130 184]
    #[110  46  79] [100  36  39] [120  56 119]
    '''[108  65  85] [98 55 45] [118  75 125]
    --[115  78  99] [105  68  59] [125  88 139]
    [119  89 104] [109  79  64] [129  99 144]'''

    #[131  64  98] [121  54  58] [141  74 138]
    #lower_rgb = np.array([105, 54, 58], dtype=np.uint8)
    #upper_rgb = np.array([141, 88, 138], dtype=np.uint8)
    
    lower_rgb = np.array([145, 84, 139], dtype=np.uint8)
    upper_rgb = np.array([165, 104, 219], dtype=np.uint8)
    # find the colors within the specified boundaries
    mask_rgb = cv2.inRange(image_edged, lower_rgb, upper_rgb)
    #cv2.imshow("mask_rgb1", mask_rgb) 
    #cv2.waitKey(0)


    # Use "close" morphological operation to close the gaps between contours
    # https://stackoverflow.com/questions/18339988/implementing-imcloseim-se-in-opencv
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)));
    #cv2.imshow("mask2", mask) 
    #cv2.waitKey(0)
    cv2.imwrite('morph_close.png',mask)

    mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)));
    #cv2.imshow("mask_rgb2", mask_rgb) 
    #cv2.waitKey(0)

    # find contours in the edge map
    cnts,hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Erase small contours, and contours which small aspect ratio (close to a square)
    for c in cnts:
        area = cv2.contourArea(c)
        print("awal",area)
        # Fill very small contours with zero (erase small contours).
        if area < 2.5:
            cv2.fillPoly(mask, pts=[c], color=0)
            continue

        # https://stackoverflow.com/questions/52247821/find-width-and-height-of-rotatedrect
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)
        print("aspect_ratio",aspect_ratio)


        # Assume zebra line must be long and narrow (long part must be at lease 1.5 times the narrow part).
        if (aspect_ratio < 1.0):
            cv2.fillPoly(mask, pts=[c], color=0)
            continue

    #cv2.imshow("mask3", mask) 
    #cv2.waitKey(0)
    cv2.imwrite('erase_small_contour.png',mask)

    #rgb contours
    # find contours in the edge map
    cnts,hier = cv2.findContours(mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Erase small contours, and contours which small aspect ratio (close to a square)
    for c in cnts:
        area = cv2.contourArea(c)
        print("area_rgb",area)
        # Fill very small contours with zero (erase small contours).
        if area < 3:
            cv2.fillPoly(mask_rgb, pts=[c], color=0)
            continue

        # https://stackoverflow.com/questions/52247821/find-width-and-height-of-rotatedrect
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)
        print("aspect_ratio",aspect_ratio)


        # Assume zebra line must be long and narrow (long part must be at lease 1.5 times the narrow part).
        if (aspect_ratio < 2.0):
            cv2.fillPoly(mask_rgb, pts=[c], color=0)
            continue

    #cv2.imshow("mask_rgb3", mask_rgb) 
    #cv2.waitKey(0)

    result = mask | mask_rgb
    # show results
    #cv2.imshow('result', result)
    #cv2.waitKey(0)


    res = cv2.bitwise_and(image,image, mask= result) 
    cv2.imwrite('res.png',np.hstack([image_edged, res]))
               

    #res = cv.dilate(res, None, iterations=1)

    #cv.imshow("mask ",mask)
    #cv2.imshow('stack2', np.hstack([image_edged, res]))
    #cv2.waitKey(0)

    mask = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)));

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] #if imutils.is_cv22() else cnts[1] 
    # loop over the contours individually
    for c in cnts:
            area = cv2.contourArea(c)
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 1.5:
                print("area_kecil", area)
                continue
         
            # compute the Convex Hull of the contour
            hull = cv2.convexHull(c) 
            if area < 40:
                print("area", area)
                cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
                counter += 1
                print("c",counter)
                cv2.putText(image_contours, "BTA {}".format(counter), (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
             
            elif 40 < area < 80:
                print("area", area)
                cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
                for i in range(2):
                      counter += 1
                      print("c",counter)
                cv2.putText(image_contours, "2 BTA bertumpuk", (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            elif 80 < area < 120:
                print("area", area)
                cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
                for i in range(3):
                      counter += 1
                      print("c",counter)
                cv2.putText(image_contours, "3 BTA bertumpuk", (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            
            elif 120 < area < 160:
                print("area", area)
                cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
                for i in range(4):
                      counter += 1
                      print("c",counter)
                cv2.putText(image_contours, "4 BTA bertumpuk", (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)      
            
            elif 160 < area < 200:
                print("area", area)
                cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
                for i in range(5):
                      counter += 1
                      print("c",counter)
                cv2.putText(image_contours, "5 BTA bertumpuk", (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)      
            
            elif 200 < area < 240:
                print("area", area)
                cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
                for i in range(6):
                      counter += 1
                      print("c",counter)
                cv2.putText(image_contours, "6 BTA bertumpuk", (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)      
            
            elif 240 < area < 280:
                print("area", area)
                cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
                for i in range(7):
                      counter += 1
                      print("c",counter)
                cv2.putText(image_contours, "7 BTA bertumpuk", (int(hull[0][0][0]), int(hull[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)      
            
    cv2.putText(image_contours, "{} BTA".format(counter), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    result = "{} sel BTA".format(counter)
    print(result)

    # Writes the output image
    filename = "output.jpg"
    print(filename)
            
    cv2.imwrite(filename,image_contours)
            
    return filename, result

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource1='../uploads/template.jpg', imagesource2='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            directory = r'C:\Users\vinaa\Python Project\Hibah 2021\IDA-Poltekkes\BTA Analyzer Web-based2\uploads'
            file_path = os.path.join(directory , filename)
            print(file_path)
            
            file.save(file_path)
            file_result, result = predict(file_path)
            
            print(result)
            print(file_result)
            file_result_path = os.path.join(directory , file_result)
             
            filename = my_random_string(6) + filename
            os.rename(file_path, os.path.join(directory, filename))
            
            file_crop = os.path.join(directory , "img_crop.png")
            file_crop_name = my_random_string(6) + "img_crop.png"
            os.rename(file_crop, os.path.join(directory, file_crop_name))
            
            filename_res = my_random_string(6) + file_result
            os.rename(file_result_path, os.path.join(directory, filename_res))
            
            #os.rename(file_result_path, os.path.join(directory , filename_result))
            
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=result, imagesource1= '../uploads/' + file_crop_name, imagesource2 = '../uploads/' + filename_res)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug.middleware.shared_data import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)