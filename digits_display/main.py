#!/usr/bin/python3

import cv2
import phase_3_display_classification.aux_functions_for_classifier as display
from phase_3_display_classification.util_interpret_digits import is_there_a_digit_display_in_this_image
import glob
import os

show_plots_to_debug = False #use True to see some images

'''
Make image binary (black and white)
'''
def binarize_image(image):
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_white_display = (0, 0, 131)
    high_white_display = (0, 0, 255)
    binary_image = cv2.inRange(HSV,low_white_display, high_white_display) #sensor green
    if show_plots_to_debug:
        cv2.imshow('window_name', binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
    return binary_image

'''
Interprete image and recognize digits
'''
def recognizeDigits(image):
    img = binarize_image(image)    
    upper_number, down_number = display.mnist_classifier(img)
    print('percentual de gas (numero de cima): %d', upper_number)
    print('ajuste de ZERO (numero de baixo): %d', down_number)
    return upper_number, down_number

'''
Check if readings are  according to Desafio Petrobras rules.
'''
def isConforme(upper_number, down_number):
    upper_ans = 'Nao conforme'
    if 45 <= upper_number <= 55:
        upper_ans = 'Conforme'

    down_ans = 'Nao conforme'
    if -5 <= down_number <= 5:
        down_ans = 'Conforme'

    return [upper_ans, down_ans]

'''
Check if they exist and detect digits in image given file name.
'''
def detect_digits_in_image(file_name):
    cv2.destroyAllWindows() 

    print('Running...')
    image = cv2.imread(file_name)
    #OpenCV uses BGR channels
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if show_plots_to_debug:
        cv2.imshow('image', image)
        cv2.waitKey(2000)

    #work with binary version of the image
    img = binarize_image(image)

    any_display = is_there_a_digit_display_in_this_image(img)
    if any_display: #True in case a display has been detected
            print('DISPLAY DETECTADO')

            upper_number, down_number = recognizeDigits(image)

            resposta = isConforme(upper_number, down_number)
            print(resposta)
    else:
            upper_number = -9999
            down_number = -9999
            print('DISPLAY NAO DETECTADO')
    return upper_number, down_number

if __name__ == '__main__':
    max_number_of_test_images = 100
    output_file_name = 'predicoes.txt'

    #provide input folder
    folder_with_PNG_files = r'C:\Users\Castro\Desktop\DesafioPetrobras\yolo_images_data\task3_digits_version10'
    #choose extension (note if upper or lowercase)
    files_with_extension = os.path.join(folder_with_PNG_files,"*.png")
    files = glob.glob(files_with_extension) #get all files with given extension
    image_counter = 1
    with open(output_file_name, 'w') as output_file:
        for file_name in files:
            print('Processing:',file_name)
            upper_number, down_number = detect_digits_in_image(file_name)
            
            output_file.write('image_' + str(image_counter) + ' ' + os.path.basename(file_name) + os.linesep)
            output_file.write(str(upper_number) + os.linesep)
            output_file.write(str(down_number) + os.linesep)
            image_counter += 1
            if image_counter > max_number_of_test_images:
                break

