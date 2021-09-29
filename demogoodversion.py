import tkinter as tk
from tkinter import *
from tkinter import filedialog
import os
import cv2
import PIL.Image
import PIL.ImageTk
import keras
import tensorflow as tf
import numpy as np
import tkinter.font as font
import json

model = keras.models.load_model("C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/modeldropout02.h5")
root=Tk()
root.geometry("1800x710")
bg=PhotoImage(file="C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/goodone.png")
my_label=Label(root,image=bg)
my_label.place(x=0,y=0)
outputLabel = ''
descriptionLabel = ''
titleLabel = ''

#for the exit button
def close_window(): 
    root.destroy()

#configure and show the global window
def confShowWindow():
    global root, titleLabel
    global recom,reg,exitt
    global imm
    
    root.title("Welcome!")
    root.configure(bg='SkyBlue1')
    
    
    titleLabel = Label(root, text = "Please insert a picture for examination",bg='#eeca93')
    titleLabel.config(font =("Courier", 14))
    titleLabel.place(x=540,y=120)
    
    brw=PhotoImage(file='C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/browse11.png')
    myFont=font.Font(family='Harlow Solid Italic',size=14)
    btn=Button(root,text="Browse \nImage",bg='#eeca93',command=lambda:predictCallback())
    btn['font']=myFont
    btn.place(x=480,y=160)
    
            
    recom=PhotoImage(file='C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/recommendations11.png')   
    button3=Button(root,image=recom,width=168, height=53,highlightthickness = 0, command=clicker)
    button3.place(x=480,y=627)
    
    imm=PhotoImage(file='C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/aaa.png')
    imgg=Label(root,image=imm)
    imgg.config(borderwidth=0)
    imgg.place(x=1000,y=590)
    
        
    reg=PhotoImage(file='C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/register11.png')
    button4=Button(root,image=reg,width=168,height=53,highlightthickness = 0, command=clicker1)
    button4.place(x=812,y=627)
    
    
    exitt=PhotoImage(file='C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/exit11.png')
    btn2=Button(root,image=exitt,width=168,height=58,highlightthickness = 0, command=close_window)
    btn2.place(x=647,y=710)
        
#predict output at button press
def predictCallback():
    global pred
    global x
    global frm
    pred=model.predict([prepare(showimage())])
    titleLabel.configure(text='')
    lb = Label(root, text = "Your result is:",bg='#eeca93')
    lb.config(font =("Courier", 14))
    lb.place(x=650,y=380)
    frm=Frame(root)
    frm.pack(side=BOTTOM,padx=20,pady=20)
    x=np.argmax(pred)
    if x==0:
        print("Detected disease: Actinic Keratosis")
        outputLabel.configure(text='Actinic Keratosis')
        outputLabel.place(x=630,y=400)
        descriptionLabel.configure(text="An actinic keratosis is a rough, \n scaly patch on the skin that develops from years of sun exposure. \n It's often found on the face, lips, ears, forearms, scalp, neck \n or back of the hands.")
    if x==1:
        print("Detected disease: Basal Cell Carcinoma")
        outputLabel.configure(text='Basal Cell Carcinoma')
        outputLabel.place(x=596,y=400)
        descriptionLabel.configure(text="Basal cell carcinoma (BCC) is the most common form of \n skin cancer and the most frequently occurring \n form of all cancers. BCCs can look like open sores, red patches, pink growths,\n shiny bumps, scars or growths with slightly elevated, rolled edges \n and/or a central indentation. \n At times, BCCs may ooze, crust, itch or bleed. The lesions commonly \n arise in sun-exposed areas of the body.")
    if x==2:
        print("Detected disease: Benign Keratosis")
        outputLabel.configure(text='Benign Keratosis')
        outputLabel.place(x=625,y=400)
        descriptionLabel.configure(text="Seborrheic keratoses are skin growths \n that some people develop as they age. They often appear on the \n back or chest, but can occur on any part of the body. Seborrheic keratoses grow slowly,\n in groups or singly. Most people will develop at least one seborrheic \n keratosis during their lifetime.")

    if x==3:
        print("Detected disease : Dermatofibroma")
        outputLabel.configure(text='Dermatofibroma')
        outputLabel.place(x=637,y=400)
        descriptionLabel.configure(text="A dermatofibroma is a common benign fibrous \n nodule usually found on the skin of the lower legs.They are sometimes attributed \n to minor trauma including insect bites, injections, or a rose thorn injury,\n  but not consistently. Multiple dermatofibromas can develop in patients \n with altered immunity such as HIV, immunosuppression,\n  or autoimmune conditions.")
        
    if x==4:
        print("Detected disease: Melanoma")
        outputLabel.configure(text='Melanoma')
        outputLabel.place(x=668,y=400)
        descriptionLabel.configure(text="Melanoma, the most serious type of skin cancer, \n develops in the cells (melanocytes) that produce melanin — the pigment that \n gives your skin its color.The exact cause of all melanomas isn't clear, but exposure \n  to ultraviolet (UV) radiation from sunlight or tanning lamps and \n beds increases your risk of developing melanoma.")
    
    if x==5:
        print("Detected disease: Melanocytic Nevus")
        outputLabel.configure(text='Melanocytic Nevus')
        outputLabel.place(x=623,y=400)
        descriptionLabel.configure(text="A melanocytic naevus, or mole, is a common \n benign skin lesion due to a local proliferation of pigment cells (melanocytes). \n A brown or black melanocytic naevus contains the pigment melanin, so may \n also be called a pigmented naevus.A melanocytic naevus can \n be present at birth (a congenital melanocytic naevus) or appear later (an acquired naevus). \n ")
        
    if x==6:
        print("Detected disease: Vascular lesion")
        outputLabel.configure(text='Vascular lesion')
        outputLabel.place(x=642,y=400)
        descriptionLabel.configure(text="Vascular lesions are relatively common abnormalities of \n the skin and underlying tissues, more commonly known as birthmarks. There are \n three major categories of vascular lesions: Hemangiomas, Vascular Malformations,\n and Pyogenic Granulomas.")
        
#for the predictions use prepare(filepath)
def prepare(filepath):
    SIZE=40
    img_array=cv2.imread(filepath)
    img_array1=cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array=cv2.resize(img_array1,(SIZE,SIZE))
    new_array1=new_array/255.
    new_array2=new_array1.reshape(-1,SIZE,SIZE,3)
    return new_array2

#ask user for image
def showimage():
    filename=filedialog.askopenfilename(parent=root,initialdir="D:/ANUL4!!!!!!!!/PNI/HAM10000/incercare/HERE",title="Select An Image",filetypes=(("JPG File","*.jpg"),("All files","*.*")))
    lbl1=Label(root,text=filename)
    img=PIL.Image.open(filename)
    resize=img.resize((250,250))
    new_img=PIL.ImageTk.PhotoImage(resize)
    lbl=Label(root)
    lbl.configure(image=new_img)
    lbl.image=new_img   
    lbl.place(x=614,y=125) 
    print(filename)
    return filename


def clicker():
    global pop
    pop=Toplevel(root)
    pop.title("Recommandation")
    pop.geometry("500x500")
    pop.config(bg="white")
    if (x==2) | (x==3) | (x==5):
        
        global bgg
        bgg=PhotoImage(file="C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/ben500.png")
        pop_label=Label(pop)
        pop_label.pack()
        my_frame1=Frame(pop)
        my_frame1.pack()    
        me_pic=Label(my_frame1,image=bgg)
        me_pic.pack()
        
        
    
    if (x==6) | (x==0):
        global bgg1
        bgg1=PhotoImage(file="C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/prem500.png")
        pop_label1=Label(pop)
        pop_label1.pack()
        my_frame2=Frame(pop)
        my_frame2.pack()    
        me_pic1=Label(my_frame2,image=bgg1)
        me_pic1.pack()
    if (x==1) :
        global bgg2
        bgg2=PhotoImage(file="C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/bcc500.png")
        pop_label2=Label(pop)
        pop_label2.pack()
        my_frame3=Frame(pop)
        my_frame3.pack()    
        me_pic2=Label(my_frame3,image=bgg2)
        me_pic2.pack()
    if (x==4) :
        global bgg3
        bgg3=PhotoImage(file="C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/DONT TOUCH/melanoma500.png")
        pop_label3=Label(pop)
        pop_label3.pack()
        my_frame4=Frame(pop)
        my_frame4.pack()    
        me_pic3=Label(my_frame4,image=bgg3)
        me_pic3.pack()

def clicker1():
    def saveinform():
        firstname_info=firstname_entry.get()
        lastname_info=lastname_entry.get()
        age_info=age_entry.get()
        age_info=str(age_info)
        review_info=review_entry.get()
        print(firstname_info, lastname_info, age_info)
        ms=Label(screen,text="You have been registered successfully!")
        ms.place(x=170,y=270)
        file=open("C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/user.txt","w")
        file.write(firstname_info)
        file.write(lastname_info)
        file.write(age_info)
        file.write(review_info)
        file.close()
        print("User", firstname_info,lastname_info,"has been registered successfully!")
        data={"first_name": firstname_info,
              "last_name": lastname_info,
              "age": age_info,
              "review":review_info
            }
        with open("C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/patients.json",'a') as f:
            json.dump(data,f,indent=2)
            f.close()
        firstname_entry.delete(0,END)
        lastname_entry.delete(0,END)
        age_entry.delete(0,END)
        review_entry.delete(0,END)
        thanks=Label(screen,text="Thank you for your time!",font="Helvetica",bg='SkyBlue1')
        thanks.place(x=150,y=400)
    screen=Tk()
    screen.title("Register")
    screen.geometry("500x500")
    heading=Label(screen,text="Registration Form",bg="#33cefa",fg="white",height=5,width=80)
    heading.pack()
    firstname_text=Label(screen,text="Firstname", )
    lastname_text=Label(screen,text="Lastname",)
    age_text=Label(screen,text="Age",)
    review_text=Label(screen,text="Review")
    firstname_text.place(x=15,y=90)
    lastname_text.place(x=15,y=150)
    age_text.place(x=15,y=210)
    review_text.place(x=250,y=90)
    
    firstname=StringVar()
    lastname=StringVar()
    age=IntVar()
    review=StringVar()
    firstname_entry=Entry(screen,textvariable=firstname,width="30")
    lastname_entry=Entry(screen,textvariable=lastname,width="30")
    age_entry=Entry(screen,textvariable=age,width="30")
    review_entry=Entry(screen,textvariable=review,width="40")
    
    firstname_entry.place(x=15,y=120)
    lastname_entry.place(x=15,y=180)
    age_entry.place(x=15,y=230)
    review_entry.place(x=250,y=120,height=50)
    
    
    register=Button(screen,text="Register",width="30",height="3",command=saveinform) 
    register.place(x=15,y=300)
    screen.mainloop()
    



confShowWindow()
outputLabel = Label(root,text="",bg='#eeca93')
outputLabel.config(font=("Hervetica",20))
descriptionLabel = Label(root,height=10,width=70,text='',justify=CENTER)
descriptionLabel.place(x=490,y=440)

root.mainloop()
