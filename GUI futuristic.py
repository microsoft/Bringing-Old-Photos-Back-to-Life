from run import main as restoration
from threading import Thread
from io import StringIO
import shutil
import os, sys, json, traceback
from tkinter import (
    Tk,
    Toplevel,
    PhotoImage,
    filedialog,
    messagebox,
    Button,
    Label,
    Frame,
    Entry,
    OptionMenu,
    Checkbutton,
    StringVar,
    BooleanVar,
    IntVar,
    scrolledtext,
    )

# empty log.txt
with open("log.txt","w") as f:
    None
    
class Root(Tk):
    def __init__(self):
        super().__init__()

        # ----------   load setting variables -------------
        
        self.with_scratch = BooleanVar(value=False)
        self.hr = BooleanVar(value=False)
        self.mystdout = ""
        self.old_stdout = ""
        self.home = os.getcwd()
        self.working_directory = "./working_directory"# directory for temporary files.
        os.makedirs(self.working_directory, exist_ok=True)
        try:
            with open("assets/settings.json", "r") as f:
                sett = json.load(f)
                
            self.gpu = sett["gpu"]
            self.output = sett["output"]
            self.selected_languaje = sett["language"]
            self.available_languages = sett["available_languages"]
            self.language(self.selected_languaje)# execute language funtion
            
        except:
            # default setting
            self.gpu = "-1"
            self.language("en")
            self.selected_languaje = "en"
            self.available_languages = ["en","es",]
            self.output = os.path.expanduser(os.path.join("~", "Documents", "Old Photo Restorer"))
        
            
        # load images
        self.bg_img = PhotoImage(file="assets/bg.png")
        self.wm_iconbitmap(bitmap="assets/icon.ico")

        
        # --------  create all buttons and frames in main menu  ----------
        
        # create main frame background
        self.main_frame = Frame(width = 406, height = 260)
        self.main_frame.pack_propagate(False)
        self.main_frame.pack()
        Label(self.main_frame, image=self.bg_img, bg="black", width = 406, height = 260).place(x=0, y=0)
        Frame(self.main_frame, bg="black", height=1).pack(pady=32)# separator
        # create loading message
        self.frame_loading = Frame(bg="#000022", width = 406, height = 260)
        Label(self.frame_loading, image=self.bg_img, width = 406,height = 260).place(x=0,y=0)
        self.loading = Label(self.frame_loading, bg="#000022", fg="light blue", font=("console",15))
        self.loading.place(x=150, y=20)
        # create console log
        self.log = scrolledtext.ScrolledText(
            self.frame_loading,
            bg="#000022",
            fg="light blue",
            width=42,
            height=9,
            highlightbackground="#00bbff",
            highlightcolor="#00bbff",
            selectbackground="gray",
            highlightthickness=1,
            )
        self.log.place(x=24, y=60)




        # button and bind effect
        b1 = Button(
            self.main_frame,
            text=self.lan["selectimages"],
            command=self.files,
            bg="#4488ee",
            activebackground="#66bbdd",
            fg="white",
            font=("Arial", 14),
            width=26,
            )
        b1.pack(anchor="center")
        def bin1(event):
            if event.type.value == str(7):
                b1["bg"] = "#44aaee"
            else:
                b1["bg"] = "#4488ee"
        b1.bind("<Enter>",bin1)
        b1.bind("<Leave>",bin1)
        
        #------------
        bar = Frame(self.main_frame)
        bar.place(x=2,y=2)
        Button(
            bar,
            text=self.lan["setting"],
            command=self.setting,
            bg="black",
            fg="white",
            activebackground="orange",
            font=("terminal", 9)
            ).grid(row=0, column=0)
        Button(
            bar,
            text=self.lan["about"],
            command=self.about,
            bg="black",
            fg="white",
            activebackground="orange",
            font=("terminal", 9)
            ).grid(row=0, column=1)
        Button(
            bar,
            text=self.lan["help"],
            command=self.help,
            bg="black",
            fg="white",
            activebackground="orange",
            font=("terminal", 9)
            ).grid(row=0, column=2)
        
        
        # box2
        box2 = Frame(self.main_frame, bg="#001122", bd=8, relief="ridge", width=300, height=100)
        box2.pack()
        box2.pack_propagate(False)
    
        
        # ------ checkbox with_scratch
        ch = Checkbutton(
            box2,
            text=self.lan["with_scratch"],
            variable=self.with_scratch,
            selectcolor="orange",
            indicatoron=True,
            background="#8899aa",
            activebackground="#77aaee",
            font=("arial",12),
            width=20,
            anchor="nw",
            )
        ch.pack(pady=6)
        def fun(e):
            if self.with_scratch.get():
                ch["bg"] = "#8899aa"
            else:
                ch["bg"] = "#00ffff"
        ch.bind("<Button-1>",fun)

        # ------ checkbox hr
        ch2 = Checkbutton(
            box2,
            text=self.lan["hr"],
            variable=self.hr,
            selectcolor="orange",
            indicatoron=True,
            background="#8899aa",
            activebackground="#77aaee",
            font=("arial",12),
            width=20,
            anchor="nw",
            )
        ch2.pack()
        def fun2(e):
            if self.hr.get():
                ch2["bg"] = "#8899aa"
            else:
                ch2["bg"] = "#00ffff"
        ch2.bind("<Button-1>",fun2)



    # ------------  Create all funtions  --------------
    
    def stop(self):
        self.stop = True
        self.destroy()
        os.abort()

    def language(self, name):
        # load built in language english
        if name == "en":
            self.lan = {
                "setting":"Setting",
                "about":"About",
                "help":"Help",
                "selectimages":"Select Images",
                "with_scratch":"with scratch",
                "hr":"High Resolution",
                "ok":"Ok",
                "reset":"Reset",
                "cancel":"Cancel",
                "programming":"<>Programming<>",
                "userinterface":"User Interface",
                "funtionandmodel":"Funtion and Models",
                "language":"language",
                "output":"Output",
                "processing":"Processing",
                "help_text":("""
    Since the model is pretrained with 256*256 images, the model may not work ideally for arbitrary resolution.

    """
                             )
                }
            
        elif name == "es":
            # load built in language spanish
            self.lan = {
                "setting":"Configuracion",
                "about":"Acerca",
                "help":"Ayuda",
                "selectimages":"Selecionar Imagenes",
                "with_scratch":"Con Rasguños",
                "hr":"Alta Resolusion",
                "ok":"Guardar",
                "reset":"Resetear",
                "cancel":"Cancelar",
                "programming":"<>Programacion<>",
                "userinterface":"Interfaz de Usuario",
                "funtionandmodel":"Funciones y Modelos",
                "language":"Idioma",
                "output":"Salida",
                "processing":"Procesando",
                "help_text":("""
    el modelo a sido entrenado con imagenes de 256*256, no se recomienda usar imagenes de alta resolusion.
    """
                             )
                }
            
        # try to load lenguage from file replace the previus loaded language
        try:
            with open(f"assets/{name}.json", "r") as f:
                l = json.load(f)
                # load only if keys are correct
                if l.keys() == self.lan.keys():
                    self.lan = l
                else:
                    messagebox.showerror("Error",f"File assets/{name}.json dictionary keys are incorrect")
        except:
            # if file not exist create new file so user can edit and customize.
            with open(f"assets/{name}.json", "w") as f:
                s = json.dumps(self.lan)
                f.write(s)



 
    def process(self):
        
        if self.with_scratch.get():
            with_scratch = "--with_scratch"
        else:
            with_scratch = ""

        if self.hr.get():
            hr = "--HR"
        else:
            hr = ""

        restoration(
            input_folder = self.working_directory + "/input",
            output_folder = self.working_directory + "/output",
            GPU = self.gpu,
            HR = hr,
            with_scratch = self.with_scratch
            )

    # show file dialog to pickup images
    def files(self):
        lista = filedialog.askopenfiles(
            filetypes=(
                "Images *.jpg",
                "Images *.jpeg",
                "Images *.png",
                "Images *.bmp",
                "All *.*",
                )
            )
        if not lista:
            return
        
        try:
            # delete old files from working directory
            w = os.walk(self.working_directory + "/output")
            for path, folder, files in w:
                for file in files:
                    os.remove(path +"/"+ file)
            l = os.listdir(self.working_directory + "/input")
            for file in l:
                os.remove(self.working_directory + "/input/" + file)

        except:
            pass
        finally:
            # make new empty working directory if not exist
            os.makedirs(self.working_directory + "/input", exist_ok=True)
            os.makedirs(self.working_directory + "/output", exist_ok=True)
            
        # copy new selected files to the working directory
        for file in lista:
            shutil.copy(file.name, self.working_directory + "/input")

        # make a funtion to use in thread
        def pro():
            self.stop = False
            self.main_frame.pack_forget()
            self.frame_loading.place(x=0,y=0)
            # ---------  redirect sys.stdout to variable to use in tkinter console --------------
            self.old_stdout = sys.stdout
            sys.stdout = self.mystdout = StringIO()

            self.loading["text"] = self.lan["processing"]
            
            # start restore process
            try:
                self.process()
            except:
                #  show error message in console and dialog, save image process log to file.
                messagebox.showerror("Error", f"Some error occurred please check log\n")
                with open("log.txt","a") as f:
                    traceback.print_exc(file=f)
                    
            # --------- restore to default sys.stdout -------------
            sys.stdout = self.old_stdout
            
            self.frame_loading.place_forget()
            self.main_frame.pack(fill="both")
            self.stop = True
            os.chdir(self.home)
            # copy the final output files to the user output folder
            files = os.listdir(self.working_directory + "/output/final_output")
            for file in files:
                shutil.copy(self.working_directory + "/output/final_output/" + file, self.output)

            
        p = Thread(target=pro)
        p.start()
        
        # insert text to the log console
        def loop():
            try:
                self.log.delete(0.0,"end")
                self.log.insert("end", self.mystdout.getvalue())
                self.log.see("end")
            except:
                None
            if self.stop:
                return
            self.timer = self.after(3000, loop)
        loop()



        
    def about(self):
        bg_color = "#666666"
        dialog = Toplevel()
        dialog.geometry(f"325x320+{self.winfo_x()+100}+{self.winfo_y()+40}")
        dialog.overrideredirect(True)
        dialog.configure(bg=bg_color, relief="ridge", bd=4)
        
        Label(dialog, text="Bringing-Old-Photos-Back-to-Life", bg=bg_color, fg="light blue", font=("Arial", 12)).pack(pady=10)
        Label(dialog, text=self.lan["programming"], bg=bg_color, fg="#00ff00", font=("Arial", 12)).pack()
        Label(dialog, text=self.lan["userinterface"], bg=bg_color, fg="#66ddff", font=("Arial", 12)).pack()
        Label(dialog, text="Erick Esau Martinez", bg=bg_color, fg="pink", font=("Arial", 12)).pack()
        Label(dialog, text=self.lan["funtionandmodel"], bg=bg_color, fg="#66ddff", font=("Arial", 12)).pack()
        Label(dialog, text="Wan, \nZiyu and Zhang, \nBo and Chen, \nDongdong and Zhang, \nPan and Chen, \nDong and Liao, \nJing and Wen, \nFang", bg=bg_color, fg="#ffdddd", font=("Arial", 12)).pack()
        
        Button(dialog, text="X", bg="red", activebackground="orange", command=dialog.destroy).place(x=290, y=0)
        dialog.grab_set()


    def help(self):
        bg_color = "#666666"
        dialog = Toplevel()
        dialog.geometry(f"220x310+{self.winfo_x()+100}+{self.winfo_y()+40}")
        dialog.overrideredirect(True)
        dialog.configure(bg=bg_color, relief="ridge", bd=4)
        Label(dialog, text = self.lan["help"], bg=bg_color, fg="#66aaff", font = ("console", 16)).pack(pady=10)
        Label(dialog, text = self.lan["help_text"], bg=bg_color, fg="white", wraplen=220).pack()
        Button(dialog, text="X", bg="red", activebackground="orange", command = dialog.destroy).place(x=190, y=0)
        dialog.grab_set()
        

    def setting(self):
        bg = "#666666"
        self.dialogo = Toplevel()
        self.dialogo.geometry(f"220x220+{self.winfo_x()+100}+{self.winfo_y()+40}")
        self.dialogo.overrideredirect(True)
        self.dialogo.configure(bg=bg, relief="ridge", bd=4)
        # frames
        box1 = Frame(self.dialogo, bd=2, relief="ridge", bg=bg)
        box1.pack(fill="x")
        box2 = Frame(self.dialogo, bd=2, relief="ridge", bg=bg)
        box2.pack(pady=6,fill="x")
        
        # create setting vars
        output = self.output
        gpu = StringVar(value=self.gpu)
        lan = StringVar(value=self.selected_languaje)
        if gpu.get() == "-1":
            gpu.set("USE CPU")

        # ------ gpu
        Label(box1, text="GPU", bg=bg,fg="white", width=14).grid(row=0, column=0, sticky="we")
        gpu_list = ['USE CPU','0', '1', '2']

        extension_menu = OptionMenu(box1, gpu, *gpu_list)
        extension_menu.grid(row=0, column=1, sticky="we")
        extension_menu.configure(
        bg=bg,
        fg="yellow",
        activebackground="orange",
        highlightbackground="green",
        highlightthickness=3,
        width=8,
        )

        # output path
        box3 = Frame(box2, bg=bg, relief="ridge", height=25)
        #box3.grid_propagate(True)
        box3.grid(sticky="we", pady=5,columnspan=2)
        def out():
            f = filedialog.askdirectory()
            if f:
                output.delete(0,last="end")
                output.insert(0,f)
                
        output = Entry(box3, font=("console",10),width=17)
        output.grid(column=0, row=0)
        output.insert(0, self.output)
        Button(box3, text=self.lan["output"], bg="light blue",width=10,font=("console",8), command=out).grid(column=1, row=0)


        # menu de lenguajes
        box4 = Frame(self.dialogo, bg=bg, relief="ridge", bd=2)
        box4.pack(fill="x")
        Label(box4, text=self.lan["language"], bg=bg,fg="white", width=14).grid(row=0, column=0)
        lan_menu = OptionMenu(box4, lan, *self.available_languages)
        lan_menu.grid(row=0, column=1)
        lan_menu.configure(
        bg=bg,
        fg="yellow",
        width=8,
        activebackground="orange",
        highlightbackground="green",
        highlightthickness=3,
        )


        def save():
            self.gpu = gpu.get()
            if self.gpu == "USE CPU":
                self.gpu = "-1"
            self.selected_languaje = lan.get()
            self.output = output.get()
            sett = {
                "gpu":self.gpu,
                "language":self.selected_languaje,
                "available_languages":self.available_languages,
                "output":self.output,
                }
            sett = json.dumps(sett)
            with open("assets/settings.json", "w") as f:
                f.write(sett)
            self.dialogo.destroy()

        def reset():
            gpu.set("USE CPU")
            output.delete(0,last="end")
            output.insert(0, os.path.expanduser(os.path.join("~", "Documents", "Old Photo Restorer")))

        Button(
            self.dialogo,
            text=self.lan["ok"],
            command=save,
            bg="#4477dd",
            fg="white",
            relief="raised",
            borderwidth=3,
            activebackground="#66aaff",
            width=8,
            ).place(x=8,y=180)
        Button(
            self.dialogo,
            text=self.lan["reset"],
            command=reset,
            bg="#4477dd",
            fg="white",
            relief="raised",
            borderwidth=3,
            activebackground="#66aaff",
            width=8,
            ).place(x=74,y=180)
        Button(
            self.dialogo,
            text=self.lan["cancel"],
            command=lambda x=None:(self.dialogo.destroy(),),
            bg="#4477dd",
            fg="white",
            relief="raised",
            borderwidth=3,
            activebackground="#66aaff",
            width=8,
            ).place(x=136,y=180)

        
        self.dialogo.grab_set()




try:
    window = Root()
    window.protocol("WM_DELETE_WINDOW", window.stop)
    window.title("Old Photo Restoration")
    window.configure(bg="gray")
    window.geometry("406x260+400+322")
    window.resizable(0,0)
    if __name__ == "__main__":
        window.mainloop()
except:
    with open("log.txt","a") as f:
        f.write("\n")
        traceback.print_exc(file=f)
        raise
