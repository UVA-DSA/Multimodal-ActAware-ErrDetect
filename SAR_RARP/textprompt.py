error_prompts = [
        "A surgeon is making multiple attempts.",
        "A surgeon is dropping or slipping the needle in the tissue.",
        "A surgeon is working with an instrument out of view.",
        "A surgeon is working with the needle out of view.",
        "A surgeon is causing tissue damage, including poor or error in tissue stabilisation.",
        "A surgeon is grasping the needle at a non-perpendicular angle.",
        "A surgeon is holding the needle incorrectly along its length.",
        "A surgeon is applying excessive force.",
        "A surgeon is not following the curve of the needle.",
        "A surgeon is inserting the needle at a non-perpendicular angle.",
        "A surgeon is grasping at the needle tip.",
        "A surgeon is loosening the suture.",
        "A surgeon is making the thread caught in the instrument.",
        "A surgeon is tying a knot that is not square.",
        "A surgeon is making an inadequate number of throws.",
        "A surgeon is pulling the suture through tissue before tying the knot.",
        "A surgeon is placing needle drives too close or too far apart.",
        "A surgeon is not pulling the suture through between needle drives.",
        "A surgeon is entangling the suture.",
        "A surgeon is fraying the suture.",
        "A surgeon is snapping the suture.",
        "A surgeon is disposing of the needle in a dangerous, poor or incorrect manner.",
        "A surgeon is working with a blurred or poor view due to incorrect camera control.",
        "A surgeon is exercising incorrect or poor instrument control, clashing, using a 3rd arm, or non-dominant use.",
        "A surgeon is executing the task perfectly."
    ]

gesture_prompts = [
"A surgeon is performing another action",                    # G0
"A surgeon is picking up the needle",                        # G1
"A surgeon is positioning the needle tip",                   # G2
"A surgeon is pushing the needle through the tissue",        # G3
"A surgeon is pulling the needle out of the tissue",         # G4
"A surgeon is tying a knot",                                 # G5
"A surgeon is cutting the suture",                           # G6
"A surgeon is returning/dropping the needle"                 # G7
]

gesture_prompts_coop = [
    "performing another action",                    # G0
    "picking up the needle",                        # G1
    "positioning the needle tip",                   # G2
    "pushing the needle through the tissue",        # G3
    "pulling the needle out of the tissue",         # G4
    "tying a knot",                                 # G5
    "cutting the suture",                           # G6
    "returning/dropping the needle"                 # G7
]

context_promtps= [
    "A surgeon is holding the needle with the left grasper",                       
    "A surgeon is holding the suture thread with the left grasper",                
    "A surgeon is holding a ring with the left grasper",                           
    "A surgeon is holding the needle with the left grasper in contact",           
    "A surgeon is holding the suture thread with the left grasper in contact",     
    "A surgeon is holding a ring with the left grasper in contact",               
    "A surgeon is holding the needle with the left grasper, not touching the fabric", 
    "A surgeon is holding the needle with the left grasper, touching the fabric",     
    "A surgeon is holding the needle with the left grasper, inserting it into the fabric", 
    "A surgeon is holding the needle with the right grasper",                       
    "A surgeon is holding the suture thread with the right grasper",                
    "A surgeon is holding a ring with the right grasper",                          
    "A surgeon is holding the needle with the right grasper in contact",           
    "A surgeon is holding the suture thread with the right grasper in contact",    
    "A surgeon is holding a ring with the right grasper in contact",                
    "A surgeon is holding the needle with the right grasper, not touching the fabric", 
    "A surgeon is holding the needle with the right grasper, touching the fabric",     
    "A surgeon is holding the needle with the right grasper, inserting it into the fabric", 
    "A surgeon is idle with the left grasper (not interacting with the needle or thread)", 
    "A surgeon is idle with the right grasper (not interacting with the needle or thread)" 
]

gesture_error_prompt = [
    # G0 - Performing another action
    "A surgeon is performing another action but makes multiple attempts.",
    "A surgeon is performing another action with an instrument out of view.",
    "A surgeon is performing another action with poor instrument control or clashing.",
    "A surgeon is performing another action with a blurred or poor view due to incorrect camera control.",
    "A surgeon is performing another action, achieving perfect execution."

    # G1 - Picking up the needle
    "A surgeon is picking up the needle but makes multiple attempts.",
    "A surgeon is picking up the needle but drops or slips the needle in the tissue.",
    "A surgeon is picking up the needle but working with the instrument out of view.",
    "A surgeon is picking up the needle but grasps the needle at a non-perpendicular angle.",
    "A surgeon is picking up the needle but holding the needle incorrectly along its length.",
    "A surgeon is picking up the needle but grasps at the needle tip.",
    "A surgeon is picking up the needle, achieving perfect execution."


    # G2 - Positioning the needle tip
    "A surgeon is positioning the needle tip but makes multiple attempts.",
    "A surgeon is positioning the needle tip but working with the needle out of view.",
    "A surgeon is positioning the needle tip but grasps the needle at a non-perpendicular angle.",
    "A surgeon is positioning the needle tip but holding the needle incorrectly along its length.",
    "A surgeon is positioning the needle tip but applying excessive force.",
    "A surgeon is positioning the needle tip, achieving perfect execution."

    # G3 - Pushing the needle through the tissue
    "A surgeon is pushing the needle through the tissue but makes multiple attempts.",
    "A surgeon is pushing the needle through the tissue but working with the needle out of view.",
    "A surgeon is pushing the needle through the tissue causing tissue damage.",
    "A surgeon is pushing the needle through the tissue but not following the curve of the needle.",
    "A surgeon is pushing the needle through the tissue but inserting the needle at a non-perpendicular angle.",
    "A surgeon is pushing the needle through the tissue but applying excessive force.",
    "A surgeon is pushing the needle through the tissue, achieving perfect execution."

    # G4 - Pulling the needle out of the tissue
    "A surgeon is pulling the needle out of the tissue but makes multiple attempts.",
    "A surgeon is pulling the needle out of the tissue but working with the needle out of view.",
    "A surgeon is pulling the needle out of the tissue but causing tissue damage.",
    "A surgeon is pulling the needle out of the tissue but fraying the suture.",
    "A surgeon is pulling the needle out of the tissue but snapping the suture.",
    "A surgeon is pulling the needle out of the tissue, achieving perfect execution."

    # G5 - Tying a knot
    "A surgeon is tying a knot that is not square.",
    "A surgeon is tying a knot but making an inadequate number of throws.",
    "A surgeon is tying a knot but entangles the suture.",
    "A surgeon is tying a knot but loosens the suture.",
    "A surgeon is tying a knot but makes multiple attempts.",
    "A surgeon is tying a knot but the thread is caught in the instrument.",
    "A surgeon is tying a knot with poor instrument control.",
    "A surgeon is tying a knot, achieving perfect execution."

    # G6 - Cutting the suture
    "A surgeon is cutting the suture but frays the suture.",
    "A surgeon is cutting the suture but snaps the suture.",
    "A surgeon is cutting the suture, achieving perfect execution."

    # G7 - Returning/dropping the needle
    "A surgeon is returning the needle but disposes of it in a dangerous, poor, or incorrect manner.",
    "A surgeon is returning the needle, achieving perfect execution."
]
lowlevelges_error_prompt  =[
       # G0
        "A surgeon is performing another action but makes multiple attempts.",
        "A surgeon is performing another action with an instrument out of view.",
        "A surgeon is performing another action with poor instrument control or clashing.",
        "A surgeon is performing another action with a blurred or poor view due to incorrect camera control.",
        "A surgeon is performing another action, achieving perfect execution."
    ,
      # G1
        "A surgeon is not holding anything then holding the needle with right grasper, needle in the fabric but makes multiple attempts.",
        "A surgeon is not holding anything then holding the needle with right grasper, needle in the fabric but drops or slips the needle in the tissue.",
        "A surgeon is not holding anything then holding the needle with right grasper, needle in the fabric but working with the instrument out of view.",
        "A surgeon is not holding anything then holding the needle with right grasper, needle in the fabric but grasps the needle at a non-perpendicular angle.",
        "A surgeon is not holding anything then holding the needle with right grasper, needle in the fabric but holding the needle incorrectly along its length.",
        "A surgeon is not holding anything then holding the needle with right grasper, needle in the fabric but grasps at the needle tip.",
        "A surgeon is not holding anything then holding the needle with right grasper, needle in the fabric, achieving perfect execution."
   ,
       # G2
        "A surgeon is holding the needle with grasper, then needle touches the tissue but makes multiple attempts.",
        "A surgeon is holding the needle with grasper, then needle touches the tissue but working with the needle out of view.",
        "A surgeon is holding the needle with grasper, then needle touches the tissue but grasps the needle at a non-perpendicular angle.",
        "A surgeon is holding the needle with grasper, then needle touches the tissue but holding the needle incorrectly along its length.",
        "A surgeon is holding the needle with grasper, then needle touches the tissue but applying excessive force.",
        "A surgeon is holding the needle with grasper, then needle touches the tissue, achieving perfect execution."
    ,
      # G3
        "A surgeon is holding the needle with grasper, then needle in the tissue but makes multiple attempts.",
        "A surgeon is holding the needle with grasper, then needle in the tissue but working with the needle out of view.",
        "A surgeon is holding the needle with grasper, then needle in the tissue causing tissue damage.",
        "A surgeon is holding the needle with grasper, then needle in the tissue but not following the curve of the needle.",
        "A surgeon is holding the needle with grasper, then needle in the tissue but inserting the needle at a non-perpendicular angle.",
        "A surgeon is holding the needle with grasper, then needle in the tissue but applying excessive force.",
        "A surgeon is holding the needle with grasper, then needle in the tissue, achieving perfect execution."
    ,
       # G4
        "A surgeon is holding the needle with grasper, then needle not touches the tissue but makes multiple attempts.",
        "A surgeon is holding the needle with grasper, then needle not touches the tissue but working with the needle out of view.",
        "A surgeon is holding the needle with grasper, then needle not touches the tissue but causing tissue damage.",
        "A surgeon is holding the needle with grasper, then needle not touches the tissue but fraying the suture.",
        "A surgeon is holding the needle with grasper, then needle not touches the tissue but snapping the suture.",
        "A surgeon is holding the needle with grasper, then needle not touches the tissue, achieving perfect execution."
    ,
       # G5
        "A surgeon is holding the thread with grasper, forming a loop, maintaining tension, and cinching the knot onto the tissue but the knot is not square.",
        "A surgeon is holding the thread with grasper, forming a loop, maintaining tension, and cinching the knot onto the tissue but makes an inadequate number of throws.",
        "A surgeon is holding the thread with grasper, forming a loop, maintaining tension, and cinching the knot onto the tissue but entangles the suture.",
        "A surgeon is holding the thread with grasper, forming a loop, maintaining tension, and cinching the knot onto the tissue but loosens the suture.",
        "A surgeon is holding the thread with grasper, forming a loop, maintaining tension, and cinching the knot onto the tissue but makes multiple attempts.",
        "A surgeon is holding the thread with grasper, forming a loop, maintaining tension, and cinching the knot onto the tissue but the thread is caught in the instrument.",
        "A surgeon is holding the thread with grasper, forming a loop, maintaining tension, and cinching the knot onto the tissue with poor instrument control.",
        "A surgeon is holding the thread with grasper, forming a loop, maintaining tension, and cinching the knot onto the tissue, achieving perfect execution."
    ,
  # G6
        "A surgeon is holding the scissors, then the scissors are closed to sever the suture but frays the suture.",
        "A surgeon is holding the scissors, then the scissors are closed to sever the suture but snaps the suture.",
        "A surgeon is holding the scissors, then the scissors are closed to sever the suture, achieving perfect execution."
    ,
       # G7
        "A surgeon is holding the needle with an instrument, moving it toward a neutral area or dropping it safely onto a surface but disposes of it in a dangerous, poor, or incorrect manner.",
        "A surgeon is holding the scissors, then the scissors are closed to sever the suture, achieving perfect execution."]

