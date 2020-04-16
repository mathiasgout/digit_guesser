# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:38:49 2020

@author: mathi
"""
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
from tensorflow import keras


class PaintApp:
    
    WIDTH = 280
    HEIGHT = WIDTH
    BONUS_HEIGHT = 30
    BORDER_WIDTH = 2
    IMAGE = Image.new("L", (HEIGHT, WIDTH), "black")
    DRAW = ImageDraw.Draw(IMAGE)
    ICON_MASTER_PATH = "C:/Users/mathi/Documents/amusement/python/digits_guesser/icon_master.ico"
    ICON_PRED_PATH = "C:/Users/mathi/Documents/amusement/python/digits_guesser/icon_pred.ico"
    MODEL_PATH = "C:/Users/mathi/Documents/amusement/python/digits_guesser/weights/model_conv2d.hdf5"
    
    
    def __init__(self):
        self.master = Tk()
        
        # Personalisation de la fenêtre
        self.master.title("Digits Guesser")
        self.master.geometry("{}x{}".format(self.WIDTH+2*self.BORDER_WIDTH, self.HEIGHT+2*(self.BONUS_HEIGHT+self.BORDER_WIDTH)))
        self.master.minsize(self.WIDTH+2*self.BORDER_WIDTH, self.HEIGHT+2*(self.BONUS_HEIGHT+self.BORDER_WIDTH))
        self.master.maxsize(self.WIDTH+2*self.BORDER_WIDTH, self.HEIGHT+2*(self.BONUS_HEIGHT+self.BORDER_WIDTH))
        self.master.iconbitmap(self.ICON_MASTER_PATH)
        
        # Message de description
        self.description = Label(self.master, text="Draw a digit between 0 and 9", font=("Helvetica", 12, "bold"),  bg="#D8EEED")
        self.description.place(x=0, y=0, height=self.BONUS_HEIGHT, width=self.WIDTH+2*self.BORDER_WIDTH)
        
        # Boutons
        self.remove_button = Button(self.master, text="\u274C", borderwidth=1, fg="red", font=("Times", 15, "bold"), command=self.delete)
        self.remove_button.place(x=0, y=self.BONUS_HEIGHT+self.HEIGHT+2*self.BORDER_WIDTH, height=self.BONUS_HEIGHT, width=self.WIDTH/2+self.BORDER_WIDTH)
        
        self.validate_button = Button(self.master, text="\u2714", borderwidth=1, fg="green", font=("Times", 15, "bold"), command=self.image_transformer)
        self.validate_button.place(x=self.WIDTH/2+self.BORDER_WIDTH, y=self.BONUS_HEIGHT+self.HEIGHT+2*self.BORDER_WIDTH, height=self.BONUS_HEIGHT, width=self.WIDTH/2+self.BORDER_WIDTH)
        
        # Zone de dessin
        self.c = Canvas(self.master, bg="white", width=self.WIDTH, height=self.HEIGHT, highlightbackground="black", highlightthickness=self.BORDER_WIDTH)
        self.c.place(x=0, y=self.BONUS_HEIGHT)
        self.c.bind("<B1-Motion>", self.paint)
        
        self.master.mainloop()
        
        
    def paint(self, event):
        """ Fonction pour dessiner """
        color = "black"
        x1, y1 = (event.x-int(self.HEIGHT/56)), (event.y-int(self.HEIGHT/56))
        x2, y2 = (event.x+int(self.HEIGHT/56)), (event.y+int(self.HEIGHT/56))
        self.c.create_oval(x1, y1, x2, y2, fill=color, outline=color)
        self.DRAW.line([x1, y1, x2, y2], fill="white", width=int(self.HEIGHT/56))
        
    def delete(self):
        """ Fonction pour supprimer le dessin et réinitialiser l'image en mémoire """
        self.c.delete("all")
        self.IMAGE = Image.new("L", (self.WIDTH, self.HEIGHT), "black")
        self.DRAW = ImageDraw.Draw(self.IMAGE)
        
    def image_transformer(self):
        """ Fonction pour transformer l'image dessinée en image 28x28 utilisable comme input """
        # On centre l'image
        boundingbox = self.IMAGE.getbbox()
        IMAGE_tmp = self.IMAGE.crop(boundingbox)
        newSize = int((5/4)*max(IMAGE_tmp.size[0], IMAGE_tmp.size[1]))
        self.IMAGE = Image.new("L", (newSize, newSize), "black") 
        self.IMAGE.paste(IMAGE_tmp, (int((newSize-IMAGE_tmp.size[0])/2), int((newSize-IMAGE_tmp.size[1])/2)))
        
        # On redimensionne l'image
        self.IMAGE_resized = self.IMAGE.resize((28,28))
        self.IMAGE_resized = np.asarray(self.IMAGE_resized)
               
        # Pour que l'intervalle de la couleur des pixels soit [0,1]
        self.IMAGE_resized = self.IMAGE_resized.astype("float32")
        self.IMAGE_resized = self.IMAGE_resized / 255
        
        self.IMAGE_resized = self.IMAGE_resized.reshape(1, 28, 28, 1)
        
        # Appel de la fonction qui donne la prediction
        self.prediction()
        
    def prediction(self):
        """ Fonction qui donne la prédiction du chiffre dessiné """
        # Importation du modèle entrainé
        self.model = keras.models.load_model(self.MODEL_PATH)
        
        # Création de la nouvelle fenêtre
        self.result_window = Tk()
        self.result_window.title("Prediction !")
        self.result_window.geometry("{}x{}".format(self.WIDTH, self.HEIGHT))
        self.result_window.minsize(self.WIDTH, self.HEIGHT)
        self.result_window.maxsize(self.WIDTH, self.HEIGHT)
        self.result_window.iconbitmap(self.ICON_PRED_PATH)
        self.result_window.config(bg="#D8EEED")
        
        
        # Affichage de la prédiction
        self.pred = self.model.predict_classes(self.IMAGE_resized)
        
        self.message = Label(self.result_window, text="You have drawn a :", font=("Helvetica", int(self.HEIGHT/19), "bold"),  bg="#D8EEED")
        self.message.place(x=0, y=self.HEIGHT/28, height=self.HEIGHT/10, width=self.WIDTH)
        
        self.pred_mes = Label(self.result_window, text = self.pred[0], font=("Helvetica", int(self.HEIGHT/3), "bold"), bg="#D8EEED", fg="red")
        self.pred_mes.place(x=0, y=self.HEIGHT/5, height=self.HEIGHT/3, width=self.WIDTH)
        
        # Affichage des probas
        self.pred_proba = self.model.predict_proba(self.IMAGE_resized)
        
        self.mes_pred_prob = Label(self.result_window, text="Predicted proba :", font=("Helvetica", int(self.HEIGHT/23), "underline bold"), bg="#D8EEED", anchor="w")
        self.mes_pred_prob.place(x=0, y=self.HEIGHT/1.7, height=self.HEIGHT/10, width=self.WIDTH/2)
        
        self.first_prob = Label(self.result_window, text="{0} : {1:.{2}f}".format((-self.pred_proba).argsort()[0][0], self.pred_proba[0][(-self.pred_proba).argsort()[0][0]], 3), font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w", fg="red")
        self.first_prob.place(x=self.WIDTH/8, y=self.HEIGHT/1.4, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        self.second_prob = Label(self.result_window, text="{0} : {1:.{2}f}".format((-self.pred_proba).argsort()[0][1], self.pred_proba[0][(-self.pred_proba).argsort()[0][1]], 3), font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        self.second_prob.place(x=5*self.WIDTH/8, y=self.HEIGHT/1.4, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        self.third_prob = Label(self.result_window, text="{0} : {1:.{2}f}".format((-self.pred_proba).argsort()[0][2], self.pred_proba[0][(-self.pred_proba).argsort()[0][2]], 3), font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        self.third_prob.place(x=self.WIDTH/8, y=self.HEIGHT/1.25, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        self.fourth_prob = Label(self.result_window, text="{0} : {1:.{2}f}".format((-self.pred_proba).argsort()[0][3], self.pred_proba[0][(-self.pred_proba).argsort()[0][3]], 3), font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        self.fourth_prob.place(x=5*self.WIDTH/8, y=self.HEIGHT/1.25, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        self.fifth_prob = Label(self.result_window, text="{0} : {1:.{2}f}".format((-self.pred_proba).argsort()[0][4], self.pred_proba[0][(-self.pred_proba).argsort()[0][4]], 3), font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        self.fifth_prob.place(x=self.WIDTH/8, y=self.HEIGHT/1.13, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        self.sixth_prob = Label(self.result_window, text="{0} : {1:.{2}f}".format((-self.pred_proba).argsort()[0][5], self.pred_proba[0][(-self.pred_proba).argsort()[0][5]], 3), font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        self.sixth_prob.place(x=5*self.WIDTH/8, y=self.HEIGHT/1.13, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        self.result_window.mainloop()
        
        
        
if __name__ == "__main__":
    paint = PaintApp()
    
    
    
        
        
        
        
        