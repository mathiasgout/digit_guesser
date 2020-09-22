import os
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras


class PaintApp:
    
    # Last model in the "models" folder
    DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    MODEL_NAME = os.listdir(DIR_PATH)[-1]
    MODEL_PATH = os.path.join(DIR_PATH, MODEL_NAME)
    print("Model used : {}".format(MODEL_PATH))

    # Model import
    model = keras.models.load_model(MODEL_PATH)
    
    def __init__(self):

        self.master = Tk()

        # Height and width
        self.screen_height = self.master.winfo_screenheight()
        self.screen_width = self.master.winfo_screenwidth()
        self.WIDTH, self.HEIGHT = int(self.screen_height * 0.4), int(self.screen_height * 0.4)
        self.BONUS_HEIGHT = int(self.screen_height * 0.03)
        self.BORDER_WIDTH = 2

        # Image config
        self.IMAGE = Image.new("L", (self.HEIGHT, self.WIDTH), "black")
        self.IMAGE_resized = Image.new("L", (self.HEIGHT, self.WIDTH), "black")
        self.DRAW = ImageDraw.Draw(self.IMAGE)

        # Personalisation de la 1ere fenêtre
        self.master.title("Digits Guesser")
        self.master.geometry("{}x{}+{}+{}".format(self.WIDTH+2*self.BORDER_WIDTH, self.HEIGHT+2*(self.BONUS_HEIGHT+self.BORDER_WIDTH),
                             int((self.screen_width-2*self.WIDTH)/2), int((self.screen_height-2*self.HEIGHT)/2)))
        self.master.minsize(self.WIDTH+2*self.BORDER_WIDTH, self.HEIGHT+2*(self.BONUS_HEIGHT+self.BORDER_WIDTH))
        self.master.maxsize(self.WIDTH+2*self.BORDER_WIDTH, self.HEIGHT+2*(self.BONUS_HEIGHT+self.BORDER_WIDTH))
        
        # Message de description
        self.description = Label(self.master,
                                 text="Draw a digit between 0 and 9",
                                 font=("Helvetica", 15, "bold"), bg="#D8EEED")
        self.description.place(x=0, y=0, height=self.BONUS_HEIGHT, width=self.WIDTH+2*self.BORDER_WIDTH)
        
        # Boutons
        self.remove_button = Button(self.master, text="REMOVE", borderwidth=1, fg="red", font=("Times", 15, "bold"),
                                    command=self.delete)
        self.remove_button.place(x=0, y=self.BONUS_HEIGHT+self.HEIGHT+2*self.BORDER_WIDTH, height=self.BONUS_HEIGHT,
                                 width=self.WIDTH/2+self.BORDER_WIDTH)
        
        self.validate_button = Button(self.master, text="VALIDATE", borderwidth=1, fg="green",
                                      font=("Times", 15, "bold"), command=self.image_transformer, state="disable")
        self.validate_button.place(x=self.WIDTH/2+self.BORDER_WIDTH,
                                   y=self.BONUS_HEIGHT+self.HEIGHT+2*self.BORDER_WIDTH, height=self.BONUS_HEIGHT,
                                   width=self.WIDTH/2+self.BORDER_WIDTH)
        
        # Zone de dessin
        self.c = Canvas(self.master, bg="white", width=self.WIDTH, height=self.HEIGHT,
                        highlightbackground="black", highlightthickness=self.BORDER_WIDTH)
        self.c.place(x=0, y=self.BONUS_HEIGHT)
        self.c.bind("<B1-Motion>", self.paint)
        
        self.master.mainloop()
        
    def paint(self, event):
        """ Fonction pour dessiner """

        self.validate_button.configure(state="normal")
        color = "black"
        x1, y1 = (event.x-int(self.HEIGHT/25)), (event.y-int(self.HEIGHT/25))
        x2, y2 = (event.x+int(self.HEIGHT/25)), (event.y+int(self.HEIGHT/25))
        self.c.create_oval(x1, y1, x2, y2, fill=color, outline=color)
        self.DRAW.line([x1, y1, x2, y2], fill="white", width=int(self.HEIGHT/25))
        
    def delete(self):
        """ Fonction pour supprimer le dessin et réinitialiser l'image en mémoire """

        # disable validate button
        self.validate_button.configure(state="disable")
        
        self.c.delete("all")
        self.IMAGE = Image.new("L", (self.WIDTH, self.HEIGHT), "black")
        self.DRAW = ImageDraw.Draw(self.IMAGE)
        
    def image_transformer(self):
        """ Fonction pour transformer l'image dessinée en image 28x28 utilisable comme input """

        # On centre l'image
        boundingbox = self.IMAGE.getbbox()
        image_tmp = self.IMAGE.crop(boundingbox)
        new_size = (int((5/4)*image_tmp.size[0]), int((5/4)*image_tmp.size[1]))
        Image.Image.paste(self.IMAGE_resized, image_tmp, (int((self.WIDTH-new_size[0])/2), int((self.WIDTH-new_size[1])/2)))
        
        # On redimensionne l'image
        self.IMAGE_resized = self.IMAGE.resize((28, 28))
        self.IMAGE_resized = np.asarray(self.IMAGE_resized)

        # Pour que l'intervalle de la couleur des pixels soit [0,1]
        self.IMAGE_resized = self.IMAGE_resized.astype("float32")
        self.IMAGE_resized = self.IMAGE_resized / 255
        
        self.IMAGE_resized = self.IMAGE_resized.reshape(1, 28, 28, 1)

        # Appel de la fonction qui donne la prediction
        self.prediction()
        
    def prediction(self):
        """ Fonction qui donne la prédiction du chiffre dessiné """
        
        # disable remove and activate button + paint
        self.validate_button.configure(state="disable")
        self.remove_button.configure(state="disable")
        self.c.unbind("<B1-Motion>")

        # Création de la nouvelle fenêtre
        self.result_window = Toplevel(self.master)
        self.result_window.protocol("WM_DELETE_WINDOW", self._state_normal)
        self.result_window.title("Prediction !")
        self.result_window.geometry("{}x{}+{}+{}".format(self.WIDTH, self.HEIGHT, int(0.6*self.WIDTH + (self.screen_width-self.WIDTH)/2), 
                                    int((self.screen_height-2*self.HEIGHT)/2)))
        self.result_window.minsize(self.WIDTH, self.HEIGHT)
        self.result_window.maxsize(self.WIDTH, self.HEIGHT)
        self.result_window.config(bg="#D8EEED")
        
        # Affichage de la prédiction
        pred = self.model.predict(self.IMAGE_resized)
        
        message = Label(self.result_window,
                        text="You have drawn a :",
                        font=("Helvetica", int(self.HEIGHT/19), "bold"),  bg="#D8EEED")
        message.place(x=0, y=self.HEIGHT/28, height=self.HEIGHT/10, width=self.WIDTH)
        
        pred_mes = Label(self.result_window,
                         text=pred[0].argmax(),
                         font=("Helvetica", int(self.HEIGHT/4), "bold"), bg="#D8EEED", fg="red")
        pred_mes.place(x=0, y=self.HEIGHT/5, height=self.HEIGHT/3, width=self.WIDTH)
        
        # Affichage des probas
        mes_pred_prob = Label(self.result_window,
                              text="Predicted proba :",
                              font=("Helvetica", int(self.HEIGHT/23), "underline bold"), bg="#D8EEED", anchor="w")
        mes_pred_prob.place(x=0, y=self.HEIGHT/1.7, height=self.HEIGHT/10, width=self.WIDTH/2)
        
        first_prob = Label(self.result_window,
                           text="{0} : {1:.{2}f}".format((-pred).argsort()[0][0], pred[0][(-pred).argsort()[0][0]], 3),
                           font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w", fg="red")
        first_prob.place(x=self.WIDTH/8, y=self.HEIGHT/1.4, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        second_prob = Label(self.result_window,
                            text="{0} : {1:.{2}f}".format((-pred).argsort()[0][1], pred[0][(-pred).argsort()[0][1]], 3),
                            font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        second_prob.place(x=5*self.WIDTH/8, y=self.HEIGHT/1.4, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        third_prob = Label(self.result_window,
                           text="{0} : {1:.{2}f}".format((-pred).argsort()[0][2], pred[0][(-pred).argsort()[0][2]], 3),
                           font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        third_prob.place(x=self.WIDTH/8, y=self.HEIGHT/1.25, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        fourth_prob = Label(self.result_window,
                            text="{0} : {1:.{2}f}".format((-pred).argsort()[0][3], pred[0][(-pred).argsort()[0][3]], 3),
                            font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        fourth_prob.place(x=5*self.WIDTH/8, y=self.HEIGHT/1.25, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        fifth_prob = Label(self.result_window,
                           text="{0} : {1:.{2}f}".format((-pred).argsort()[0][4], pred[0][(-pred).argsort()[0][4]], 3),
                           font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        fifth_prob.place(x=self.WIDTH/8, y=self.HEIGHT/1.13, height=self.HEIGHT/15, width=3*self.WIDTH/8)
        
        sixth_prob = Label(self.result_window,
                           text="{0} : {1:.{2}f}".format((-pred).argsort()[0][5], pred[0][(-pred).argsort()[0][5]], 3),
                           font=("Helvetica", int(self.HEIGHT/28), "bold"), bg="#D8EEED", anchor="w")
        sixth_prob.place(x=5*self.WIDTH/8, y=self.HEIGHT/1.13, height=self.HEIGHT/15, width=3*self.WIDTH/8)

        # Reinitialisation
        self.IMAGE_resized = Image.new("L", (self.HEIGHT, self.WIDTH), "black")
        self.IMAGE = Image.new("L", (self.HEIGHT, self.WIDTH), "black")

    def _state_normal(self):
        """ Enable validate button + paint """

        self.result_window.destroy()

        self.remove_button.configure(state="normal")
        
        # delete draw
        self.c.delete("all")
        self.IMAGE = Image.new("L", (self.WIDTH, self.HEIGHT), "black")
        self.DRAW = ImageDraw.Draw(self.IMAGE)
        self.c.bind("<B1-Motion>", self.paint)


if __name__ == "__main__":
    PaintApp()
