import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import threading

# Inicializar los módulos de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

class ObjectronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe Shoe/Cup 3D  Detector")
        self.root.geometry("600x600")
        self.root.configure(bg="#c2f0c2")  # Color de fondo verde claro
        
        # Título centrado con instrucciones
        title_label = tk.Label(self.root, text="MediaPipe Shoe/Cup 3D Detector", font=("Arial", 20), bg="#c2f0c2")
        title_label.pack(pady=20)
        
        instructions_label = tk.Label(self.root, text="Seleccione un modelo.Luego haga clic en 'Encender cámara' para comenzar la detección.", font=("Arial", 12), bg="#c2f0c2")
        instructions_label.pack(pady=10)
        
        # Botón para cerrar la aplicación
        close_button = tk.Button(self.root, text="\u274C Cerrar cámara", command=self.close_camera)
        close_button.pack(pady=10)
        
        # Botón para encender la cámara
        self.camera_button = tk.Button(self.root, text="\u23F5 Encender cámara", command=self.start_camera)
        self.camera_button.pack(pady=10)
        
        # Radio buttons para seleccionar el modelo de objeto
        self.model_name = tk.StringVar()
        self.models_frame = tk.Frame(self.root, bg="#c2f0c2")  # Ahora es un atributo de la instancia
        self.models_frame.pack(pady=10)
        
        models = ["Cup", "Shoe"]
        self.radio_buttons = []
        for model in models:
            rb = tk.Radiobutton(self.models_frame, text=model, variable=self.model_name, value=model)
            rb.pack(anchor=tk.W)
            self.radio_buttons.append(rb)
        
        # Botón para reiniciar valores
        reset_button = tk.Button(self.root, text="\u21BA Reiniciar valores", command=self.reset_values)
        reset_button.pack(pady=10)
        
        # Configurar MediaPipe Objectron
        self.objectron = None
        self.video_thread = None
    
    def close_camera(self):
        if self.objectron:
            self.objectron.close()
            self.objectron = None
        cv2.destroyAllWindows()
    
    def start_camera(self):
        selected_model = self.model_name.get()
        
        if not selected_model:
            messagebox.showwarning("Alerta", "Selecciona un modelo de objeto primero.")
            return
        
        # Configurar Objectron para detectar objetos en video en tiempo real
        self.objectron = mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=5,
            min_detection_confidence=0.5,
            model_name=selected_model
        )
        
        # Habilitar los radio buttons después de iniciar la cámara
        for rb in self.radio_buttons:
            rb.config(state=tk.NORMAL)
        
        # Iniciar un hilo para la captura de video
        self.video_thread = threading.Thread(target=self.run_video_capture)
        self.video_thread.start()
    
    def run_video_capture(self):
        # Iniciar la captura de video desde la cámara web
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara.")
            return
        
        # Definir el tamaño deseado para la ventana de video
        window_width = 400
        window_height = 400
        
        # Configurar la ventana de visualización
        cv2.namedWindow('Objectron', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Objectron', window_width, window_height)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "No se pudo capturar el frame.")
                break
            
            # Redimensionar el frame al tamaño deseado (400x400)
            frame_resized = cv2.resize(frame, (window_width, window_height))
            
            # Convertir la imagen de BGR a RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Procesar la imagen con Objectron
            results = self.objectron.process(frame_rgb)
            
            # Dibujar los resultados en el frame
            if results.detected_objects:
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(
                        frame_resized,
                        detected_object.landmarks_2d,
                        mp_objectron.BOX_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=6),
                        mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2)
                    )
                    mp_drawing.draw_axis(
                        frame_resized,
                        detected_object.rotation,
                        detected_object.translation
                    )
            
            # Mostrar el frame redimensionado con los dibujos
            cv2.imshow('Objectron', frame_resized)
            
            # Salir del bucle al presionar la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Liberar la captura de video y cerrar todas las ventanas
        cap.release()
        cv2.destroyAllWindows()
        
        # Detener Objectron
        if self.objectron:
            self.objectron.close()
            self.objectron = None
            
        # Desactivar los radio buttons al cerrar la cámara
        for rb in self.radio_buttons:
            rb.config(state=tk.DISABLED)
    
    def reset_values(self):
        self.model_name.set("")  # Reiniciar la selección del modelo

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectronApp(root)
    root.mainloop()
