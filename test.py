# col1, col2 = st.columns([1,1])
# Placeholder for live feed
# frame_placeholder11 = col1.empty()
# frame_placeholder12 = col2.empty()
# frame_placeholder21 = col1.empty()
# frame_placeholder22 = col2.empty()


# h, w = frame.shape[:2]      # h=480, w=640

# half_w = w // 2             # 320
# half_h = h // 2             # 240

# frame1 = frame[0:half_h, 0:half_w]
# frame2 = frame[0:half_h, half_w:w]           # Top-Right
# frame3 = frame[half_h:h, 0:half_w]           # Bottom-Left
# frame4 = frame[half_h:h, half_w:w]  

# frame_placeholder11.image(frame1, channels="RGB")
# frame_placeholder12.image(frame2, channels="RGB")
# frame_placeholder21.image(frame3, channels="RGB")
# frame_placeholder22.image(frame4, channels="RGB")



        # if counter%100==0:
        #     first_corner[0]+=x_add
        #     second_corner[0]+=x_add
        #     if second_corner[0] >= frame_size[0]:
        #         first_corner[1]+=y_add
        #         second_corner[1]+=y_add
        #         first_corner[0] = 0
        #         second_corner[0] = 64
        #     if second_corner[1] >= frame_size[1]:
        #         first_corner = [0, 0]
        #         second_corner = [64, 48]