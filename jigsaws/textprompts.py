error_prompt = [
    "The surgeon is making multiple attempts.", #1  
    "The surgeon is dropping the needle.", #2
    "The surgeon is losing the needle out of view.", #3
    "The surgeon is executing the task perfectly.", #4
    "The surgeon is not moving the needle along the needle curve.", #5
    "The surgeon is grasping the needle too far or too short.", #6  
    "The surgeon is holding the needle not perpendicular to the tissue." #7 
]


Context_Prompt = [
    "A surgeon is holding the needle with the left grasper",                        # 0
    "A surgeon is holding the suture thread with the left grasper",                # 1
    "A surgeon is holding a ring with the left grasper",                           # 2
    "A surgeon is holding the needle with the left grasper in contact",            # 3
    "A surgeon is holding the suture thread with the left grasper in contact",     # 4
    "A surgeon is holding a ring with the left grasper in contact",                # 5
    "A surgeon is holding the needle with the left grasper, not touching the fabric", # 6
    "A surgeon is holding the needle with the left grasper, touching the fabric",     # 7
    "A surgeon is holding the needle with the left grasper, inserting it into the fabric", # 8
    "A surgeon is holding the needle with the left grasper, not touching a ring",      # 9
    "A surgeon is holding the needle with the left grasper, touching a ring",         # 10
    "A surgeon is holding the needle with the left grasper, inserting it into a ring",# 11
    "A surgeon is holding the needle with the right grasper",                       # 12
    "A surgeon is holding the suture thread with the right grasper",                # 13
    "A surgeon is holding a ring with the right grasper",                           # 14
    "A surgeon is holding the needle with the right grasper in contact",            # 15
    "A surgeon is holding the suture thread with the right grasper in contact",     # 16
    "A surgeon is holding a ring with the right grasper in contact",                # 17
    "A surgeon is holding the needle with the right grasper, not touching the fabric", # 18
    "A surgeon is holding the needle with the right grasper, touching the fabric",     # 19
    "A surgeon is holding the needle with the right grasper, inserting it into the fabric", # 20
    "A surgeon is holding the needle with the right grasper, not touching a ring",      # 21
    "A surgeon is holding the needle with the right grasper, touching a ring",         # 22
    "A surgeon is holding the needle with the right grasper, inserting it into a ring",# 23
    "A surgeon is idle with the left grasper (not interacting with the needle or thread)", # 24
    "A surgeon is idle with the right grasper (not interacting with the needle or thread)" # 25
]

gesture_error_prompt=[
    "A surgeon is reaching for the needle with the right hand and makes multiple attempts.",
    "A surgeon is reaching for the needle with the right hand, but the needle drops.",
    "A surgeon is reaching for the needle with the right hand, but the needle is out of view.",
    "A surgeon is reaching for the needle with the right hand, achieving perfect execution.",
    "A surgeon is positioning the needle but makes multiple attempts.",
    "A surgeon is positioning the needle, but the needle drops.",
    "A surgeon is positioning the needle, but the needle is out of view.",
    "A surgeon is positioning the needle, achieving perfect execution.",
    "A surgeon is pushing the needle through the tissue, but the needle is not moving along the needle curve.",
    "A surgeon is pushing the needle through the tissue but makes multiple attempts.",
    "A surgeon is pushing the needle through the tissue, but the needle drops.",
    "A surgeon is pushing the needle through the tissue, but the needle is out of view.",
    "A surgeon is pushing the needle through the tissue, achieving perfect execution.",
    "A surgeon is transferring the needle from left to right but makes multiple attempts.",
    "A surgeon is transferring the needle from left to right but grasps too far/too short on the needle.",
    "A surgeon is transferring the needle from left to right, but the needle drops.",
    "A surgeon is transferring the needle from left to right, but the needle is out of view.",
    "A surgeon is transferring the needle from left to right, achieving perfect execution.",
    "A surgeon is moving to the center with the needle in grip, but the needle drops.",
    "A surgeon is moving to the center with the needle in grip, but the needle is out of view.",
    "A surgeon is moving to the center with the needle in grip, achieving perfect execution.",
    "A surgeon is pulling the suture with the left hand but makes multiple attempts.",
    "A surgeon is pulling the suture with the left hand, but the needle drops.",
    "A surgeon is pulling the suture with the left hand, but the needle is out of view.",
    "A surgeon is pulling the suture with the left hand, achieving perfect execution.",
    "A surgeon is orienting the needle but makes multiple attempts.",
    "A surgeon is orienting the needle, but it is not perpendicular to the tissue.",
    "A surgeon is orienting the needle, but the needle drops.",
    "A surgeon is orienting the needle, but the needle is out of view.",
    "A surgeon is orienting the needle, achieving perfect execution.",
    "A surgeon is using the right hand to help tighten the suture but makes multiple attempts.",
    "A surgeon is using the right hand to help tighten the suture, but the needle drops.",
    "A surgeon is using the right hand to help tighten the suture, but the needle is out of view.",
    "A surgeon is using the right hand to help tighten the suture, achieving perfect execution."
]
gesture_prompt = [
"A surgeon is performing another action", #1
"A surgeon is reaching for needle with right hand", #2
"A surgeon is positioning needle", #3
"A surgeon is pushing needle through tissue", #4
"A surgeon is transferring needle from left to right", #5
"A surgeon is moving to center with needle in grip", #6
"A surgeon is pulling suture with left hand", #7
"A surgeon is orienting needle", #8
"A surgeon is using right hand to help tighten suture", #9

]

lowlevel_gesture_error_prompt=[
    "A surgeon is holding the needle with the right grasper but makes multiple attempts.",
    "A surgeon is holding the needle with the right grasper but the needle drops.",
    "A surgeon is holding the needle with the right grasper but the needle is out of view.",
    "A surgeon is holding the needle with the right grasper, achieving perfect execution.",
    "A surgeon is holding the needle with the right grasper to touch the fabric or ring but makes multiple attempts.",
    "A surgeon is holding the needle with the right grasper to touch the fabric or ring but the needle drops.",
    "A surgeon is holding the needle with the right grasper to touch the fabric or ring but the needle is out of view.",
    "A surgeon is holding the needle with the right grasper to touch the fabric or ring, achieving perfect execution.",
    "A surgeon is holding the needle with the right grasper to push the needle into the fabric or ring but the needle is not moving along the needle curve.",
    "A surgeon is holding the needle with the right grasper to push the needle into the fabric or ring but makes multiple attempts.",
    "A surgeon is holding the needle with the right grasper to push the needle into the fabric or ring but the needle drops.",
    "A surgeon is holding the needle with the right grasper to push the needle into the fabric or ring but the needle is out of view.",
    "A surgeon is holding the needle with the right grasper to push the needle into the fabric or ring, achieving perfect execution.",
    "A surgeon is holding the needle with the right grasper from the left grasper but makes multiple attempts.",
    "A surgeon is holding the needle with the right grasper from the left grasper but grasps too far or too short on the needle.",
    "A surgeon is holding the needle with the right grasper from the left grasper but the needle drops.",
    "A surgeon is holding the needle with the right grasper from the left grasper but the needle is out of view.",
    "A surgeon is holding the needle with the right grasper from the left grasper, achieving perfect execution.",
    "A surgeon is holding the needle with the right grasper to contact the left grasper but the needle drops.",
    "A surgeon is holding the needle with the right grasper to contact the left grasper but the needle is out of view.",
    "A surgeon is holding the needle with the right grasper to contact the left grasper, achieving perfect execution.",
    "A surgeon is holding the needle with the right grasper to release contact with the fabric or ring but makes multiple attempts.",
    "A surgeon is holding the needle with the right grasper to release contact with the fabric or ring but the needle drops.",
    "A surgeon is holding the needle with the right grasper to release contact with the fabric or ring but the needle is out of view.",
    "A surgeon is holding the needle with the right grasper to release contact with the fabric or ring, achieving perfect execution.",
    "A surgeon is holding the needle with the right grasper while switching between the left grasper but makes multiple attempts.",
    "A surgeon is holding the needle with the right grasper while switching between the left grasper but it is not perpendicular to the grasper.",
    "A surgeon is holding the needle with the right grasper while switching between the left grasper but the needle drops.",
    "A surgeon is holding the needle with the right grasper while switching between the left grasper but the needle is out of view.",
    "A surgeon is holding the needle with the right grasper while switching between the left grasper, achieving perfect execution.",
    "A surgeon is holding the needle with the left grasper to let the right grasper contact the thread but makes multiple attempts.",
    "A surgeon is holding the needle with the left grasper to let the right grasper contact the thread but the needle drops.",
    "A surgeon is holding the needle with the left grasper to let the right grasper contact the thread but the needle is out of view.",
    "A surgeon is holding the needle with the left grasper to let the right grasper contact the thread, achieving perfect execution."
]