sat_list = ['S2', 'S1']
prepost = ['pre', 'post']
stacking = False

image_list = []
for sat in (sat_list):
    # post_fps = self.fps_dict[sat][1]
    # image_post = tiff.imread(post_fps[i]) # C*H*W
    # image_post = np.nan_to_num(image_post, 0)
    # if sat in ['S1', 'ALOS']: image_post = (np.clip(image_post, -30, 0) + 30) / 30
    # image_post = image_post[self.band_index_dict[sat],] # select bands

    image_post = f"{sat}_post"

    if 'pre' in prepost:
        # pre_fps = self.fps_dict[sat][0]
        # image_pre = tiff.imread(pre_fps[i])
        # image_pre = np.nan_to_num(image_pre, 0)
        # if sat in ['S1', 'ALOS']: image_pre = (np.clip(image_pre, -30, 0) + 30) / 30
        # image_pre = image_pre[self.band_index_dict[sat],] # select bands
        image_pre = f"{sat}_pre"
        
        if stacking: # if stacking bi-temporal data
            # stacked = np.concatenate((image_pre, image_post), axis=0) 
            stacked = f"{image_pre}_{image_post}"
            image_list.append(stacked) #[x1, x2]
        else:
            image_list += [image_pre, image_post] #[t1, t2]
    else:
        image_list.append(image_post) #[x1_t2, x2_t2]

print(image_list)