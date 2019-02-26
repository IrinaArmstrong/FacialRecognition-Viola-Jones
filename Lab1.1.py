import cv2 as cv

# drHouse.jpg
image_filename = "example_2.jpg"
image = cv.imread(image_filename, cv.IMREAD_COLOR)

if (image is None):  # Checkn for invalid input
    print("Could not open or find the image!")
else:
    print("Size of image: ", image.shape)  # print size of image
    # cv.namedWindow("Image", cv.WINDOW_NORMAL)
    # Convert colored images to grayscale
    grayscale_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # Improves the contrast of an image, in order to stretch out the intensty range
    grayscale_image = cv.equalizeHist(grayscale_image)
    # Load the required XML classifier
    # "haarcascade_frontalface_alt.xml" or "haarcascade_frontalface_default.xml"
    cascade_filename = "haarcascade_frontalface_default.xml"
    cascade = cv.CascadeClassifier(cascade_filename)

    # Detects objects of different sizes in the input image.
    # The detected objects are returned as a list of rectangles.
    # Parameters:
    #           cascade – Haar classifier cascade.
    #           image – Matrix containing an image where objects are detected.
    #           scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
    #           minNeighbors – Parameter specifying how many neighbors each candidate
    #                           rectangle should have to retain it.
    #           flags – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects.
    #           minSize – Minimum possible object size. Objects smaller than that are ignored.
    #           maxSize – Maximum possible object size. Objects larger than that are ignored.
    faces = cascade.detectMultiScale(grayscale_image, scaleFactor=1.1,
                                     minNeighbors=3, minSize=(20, 20))
    if len(faces) == 0:
        print("No face detected!")
    else:
        # Rectangles as: x1, y1, w(width), h(height) convert to -> x1, y1, x2, y2
        faces[:, 2:] += faces[:, :2]
        print("Faces detected: ", len(faces))
        copy_image = image.copy()
        for x1, y1, x2, y2 in faces:
            cv.rectangle(copy_image, (x1, y1), (x2, y2), (255, 0, 123), 2)
            # Create window for display
            cv.namedWindow("Face detected", cv.WINDOW_NORMAL)
            cv.imshow("Face detected", copy_image)
    # cv.imshow("Grayscale Image", grayscale_image)  # Show grayscale image in the window
    # cv.imshow("Image", image)  # Create window for display
    cv.waitKey(0)  # Wait for key
    cv.destroyAllWindows()

