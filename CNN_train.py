#coding: utf-8
import numpy as np
import sys

# 任意输入的单通道图像和单通道卷积核
def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    #遍历图像进行卷积
    for r in np.uint16(np.arange(filter_size/2, img.shape[0]-filter_size/2)):
        for c in np.uint16(np.arange(filter_size/2, img.shape[1]-filter_size/2)):
            curr_region = img[r-np.uint16(np.floor(filter_size/2)): r+np.uint16(np.ceil(filter_size/2)),
                          c-np.uint16(np.floor(filter_size / 2)): c+np.uint16(np.ceil(filter_size/2))]
            curr_result = curr_region*conv_filter
            conv_sum = np.sum(curr_result)
            result[r,c] = conv_sum

    final_result = result[np.uint16(filter_size/2):result.shape[0]-np.uint16(filter_size/2),
                   np.uint16(filter_size/2):result.shape[1]-np.uint16(filter_size/2)]
    return final_result

#color image:img=(r,c,channels) ; conv_filter=(k,F,F,channels)
# gray image: img=(r,c) ; conv_filter=(k,F,F)
def conv(img, conv_filter):
    if len(img.shape)>2 or len(conv_filter.shape)>3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print("Error: Filter must be a square matrix, I.e. number of rows and columns must be match.")
        sys.exit()
    if conv_filter.shape[1]%2 == 0:
        print ("Error: filter must have an odd size, I.e. number of rows and columns must be odd!")
        sys.exit()

    # n=(N-F+2P)/S+1
    feature_map = np.zeros((img.shape[0]-conv_filter.shape[1]+1,img.shape[1]-conv_filter.shape[1]+1,
                            conv_filter.shape[0]))

    for filter_num in range(conv_filter.shape[0]):
        print("Filter: ", filter_num +1)
        curr_filter = conv_filter[filter_num,:]
        # print (curr_filter)

        #判断是否为多通道卷积核
        if len(curr_filter.shape)>2:
            print ("number of filter: {}".format(len(curr_filter.shape)))
            conv_map = conv_(img[:,:,0], curr_filter[:,:,0])
            # 对多个通道的卷积结构进行求和
            for ch_num in range(1,curr_filter.shape[-1]):
                conv_map = conv_map +conv_(img[:,:,ch_num], curr_filter[:,:,ch_num])
        else:
            conv_map = conv_(img,curr_filter)

        feature_map[:,:,filter_num] = conv_map

    return feature_map


# 池化层
def pooling(feature_map, size=2, stride=2):
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size)/stride+1),
                        np.uint16((feature_map.shape[1]-size)/stride+1),
                        feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2=0
        for r in np.arange(0,feature_map.shape[0]-size-1, stride):
            c2=0
            for c in np.arange(0,feature_map.shape[1]-size-1,stride):
                pool_out[r2,c2,map_num] = np.max(feature_map[r:r+size,c:c+size])
                c2 = c2+1
            r2 =r2+1
    print ("size of pool_out:{}", pool_out.shape)
    return pool_out

# def pooling(feature_map, size=2, stride=2):
#     #Preparing the output of the pooling operation.
#     pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride),
#                             np.uint16((feature_map.shape[1]-size+1)/stride),
#                             feature_map.shape[-1]))
#     for map_num in range(feature_map.shape[-1]):
#         r2 = 0
#         for r in np.arange(0,feature_map.shape[0]-size-1, stride):
#             c2 = 0
#             for c in np.arange(0, feature_map.shape[1]-size-1, stride):
#                 pool_out[r2, c2, map_num] = np.max(feature_map[r:r+size,  c:c+size])
#                 c2 = c2 + 1
#             r2 = r2 +1
#     print("size of pool_out:{}", pool_out.shape)
#     return pool_out

# 激活函数
def relu(feature_map):
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0,feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r,c,map_num] = np.max(feature_map[r,c,map_num],0)

    return relu_out
