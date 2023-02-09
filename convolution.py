import cv2
import numpy as np
import torchvision.transforms as T


# Python 3 program to perform 2D convolution operation
import torch
import torch.nn as nn



for img_id in range (1):
    #lettura delle immagine
    img = cv2.imread('C:\\Users\\tsarr\\Desktop\\Computer Vision\\dataset\\KITTI_odometry_gray\\00\\image_l\\' +str(img_id).zfill(6) + '.png', 0)
    #print(img.shape)
    print("This is our input tensor for image id: " + str(img_id) + " with shape: " + str(img.shape))
    print(img)

    #print("This is our output image for image id: " + str(img_id))
    cv2.imshow("Here is image:", img)
    cv2.waitKey(5)

    #transform2 for square resize
    transform2 = T.Resize(size=(376,376))
    #transform to tensor
    transform = T.ToTensor()

    img_to_resized = transform(img)

    input = transform2(img_to_resized)

    '''torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    '''
    in_channels = input.shape[0]
    out_channels = 3
    kernel_size = 3
    conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    # conv = nn.Conv2d(2, 3, 2)

    '''input of size [N,C,H, W]
    N==>batch size,
    C==> number of channels,
    H==> height of input planes in pixels,
    W==> width in pixels.
    '''

    # define the input with below info
    N = 2
    C = input.shape[0]
    H = input.shape[1]
    W = input.shape[2]
    #input = torch.empty(N, C, H, W).random_(256)
    print("Input Tensor:", input)
    print("Input Size:", input.size())

    # Perform convolution operation
    output = conv(input)
    print("Output Tensor:", output)
    print("Output Size:", output.size())

    input_1 = output
    C = input_1.shape[0]
    H = input_1.shape[1]
    W = input_1.shape[2]

    # With square kernels (2,2) and equal stride
    conv = nn.Conv2d(C, 3, 3)

    output = conv(input_1)
    print("Output Tensor:", output)
    print("Output Size:", output.size())


    for i in range (376):
        if(output.shape[1] > 224):
            input_1 = output
            C = input_1.shape[0]
            H = input_1.shape[1]
            W = input_1.shape[2]

            # With square kernels (2,2) and equal stride
            conv = nn.Conv2d(C, 3, 3)

            output = conv(input_1)
            #print("Output Tensor:", output)
            #print("Output Size:", output.size())


    print("Final Output Tensor:", output)
    print("Final output size:", output.size())

    out = output

    print("This is our output tensor for image id: " + str(img_id))
    print(out)

    #trasformare il tensore ad un immagine
    transform1 = T.ToPILImage()
    img_out = transform1(out)

    img_out.show()