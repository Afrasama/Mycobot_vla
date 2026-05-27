"""
GUI Status Display for Robot Actions
Provides real-time visual feedback in a separate tkinter window
"""
import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Dict, List, Optional


class GUIStatusDisplay:
    def __init__(self):
        self.current_status = "IDLE"
        self.action_details = ""
        self.attempt_count = 0
        self.distance_to_goal = 0.0
        self.llm_decision = ""
        
        # Robot joint angles
        self.joint_angles = {}
        self.show_joint_angles = False
        
        # Create GUI in separate thread
        self.root = None
        self.status_label = None
        self.action_label = None
        self.attempt_label = None
        self.distance_label = None
        self.llm_label = None
        self.joint_angles_label = None
        self.joint_angles_frame = None
        self.gui_thread = None
        self.running = True
        
        # Status colors
        self.colors = {
            "IDLE": "#808080",      # Gray
            "PLANNING": "#0000FF",   # Blue
            "APPROACHING": "#FF8000", # Orange
            "GRASPING": "#FF0000",    # Red
            "GRASPED": "#00FF00",    # Green
            "LIFTING": "#00FFFF",     # Cyan
            "PLACING": "#FF00FF",     # Magenta
            "PLACED": "#00CC00",      # Dark Green
            "REFLECTING": "#FFFF00",  # Yellow
            "VLA RECOVERY": "#FF6600",  # Bright orange
            "FAILED": "#CC0000",      # Dark Red
        }
        
        # Start GUI in separate thread
        self.start_gui_thread()
    
    def start_gui_thread(self):
        """Start the GUI in a separate thread"""
        try:
            self.gui_thread = threading.Thread(target=self.create_gui, daemon=True)
            self.gui_thread.start()
            
            # Wait a bit for GUI to initialize
            time.sleep(0.2)
            print("GUI thread started successfully")
        except Exception as e:
            print(f"Failed to start GUI thread: {e}")
            self.running = False
    
    def create_gui(self):
        """Create the tkinter GUI window"""
        self.root = tk.Tk()
        self.root.title("Robot Status Monitor")
        self.root.geometry("400x300")
        self.root.configure(bg="#1e1e1e")
        
        # Make window stay on top
        self.root.attributes("-topmost", True)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="#1e1e1e", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="ROBOT STATUS MONITOR", 
            font=("Arial", 16, "bold"),
            fg="white", 
            bg="#1e1e1e"
        )
        title_label.pack(pady=(0, 20))
        
        # Status frame
        status_frame = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_label = tk.Label(
            status_frame,
            text="IDLE",
            font=("Arial", 14, "bold"),
            fg=self.colors["IDLE"],
            bg="#2d2d2d",
            pady=10
        )
        self.status_label.pack()
        
        # Action label
        self.action_label = tk.Label(
            main_frame,
            text="Action: -",
            font=("Arial", 11),
            fg="#cccccc",
            bg="#1e1e1e",
            anchor="w"
        )
        self.action_label.pack(fill=tk.X, pady=5)
        
        # Metrics frame
        metrics_frame = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.RAISED, bd=2)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # Attempt label
        self.attempt_label = tk.Label(
            metrics_frame,
            text="Attempt: 0",
            font=("Arial", 10),
            fg="#cccccc",
            bg="#2d2d2d",
            anchor="w",
            padx=10,
            pady=5
        )
        self.attempt_label.pack(fill=tk.X)
        
        # Distance label
        self.distance_label = tk.Label(
            metrics_frame,
            text="Distance: N/A",
            font=("Arial", 10),
            fg="#cccccc",
            bg="#2d2d2d",
            anchor="w",
            padx=10,
            pady=5
        )
        self.distance_label.pack(fill=tk.X)
        
        # Joint Angles frame (initially hidden)
        self.joint_angles_frame = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.RAISED, bd=2)
        
        # Joint Angles title
        joint_title = tk.Label(
            self.joint_angles_frame,
            text="ROBOT JOINT ANGLES",
            font=("Arial", 11, "bold"),
            fg="#00ffff",
            bg="#2d2d2d",
            pady=5
        )
        joint_title.pack()
        
        # Joint Angles label
        self.joint_angles_label = tk.Label(
            self.joint_angles_frame,
            text="Waiting for robot to start...",
            font=("Courier", 9),
            fg="#cccccc",
            bg="#2d2d2d",
            anchor="nw",
            justify=tk.LEFT,
            padx=10,
            pady=5,
            wraplength=350
        )
        self.joint_angles_label.pack(fill=tk.BOTH, expand=True)
        
        # LLM frame
        llm_frame = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.RAISED, bd=2)
        llm_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # LLM label
        self.llm_label = tk.Label(
            llm_frame,
            text="LLM: -",
            font=("Arial", 10),
            fg="#ffff00",
            bg="#2d2d2d",
            anchor="nw",
            justify=tk.LEFT,
            padx=10,
            pady=5,
            wraplength=350
        )
        self.llm_label.pack(fill=tk.BOTH, expand=True)
        
        # Start GUI update loop
        self.update_gui()
        
        # Run the GUI
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def update_gui(self):
        """Update GUI elements with current data"""
        if self.root and self.running:
            try:
                # Update status
                if self.status_label:
                    self.status_label.config(
                        text=self.current_status,
                        fg=self.colors.get(self.current_status, "#808080")
                    )
                
                # Update action
                if self.action_label:
                    action_text = f"Action: {self.action_details}" if self.action_details else "Action: -"
                    self.action_label.config(text=action_text)
                
                # Update metrics
                if self.attempt_label:
                    self.attempt_label.config(text=f"Attempt: {self.attempt_count}")
                
                if self.distance_label:
                    distance_str = f"{self.distance_to_goal:.3f}m" if self.distance_to_goal is not None else "N/A"
                    self.distance_label.config(text=f"Distance: {distance_str}")
                
                # Update joint angles
                if self.joint_angles_label and self.show_joint_angles:
                    if self.joint_angles:
                        angles_text = "Joint Angles (degrees):\n"
                        for joint_name, angle in self.joint_angles.items():
                            angles_text += f"{joint_name:12s}: {angle:8.2f}°\n"
                        self.joint_angles_label.config(text=angles_text)
                    else:
                        self.joint_angles_label.config(text="Reading joint angles...")
                
                # Update LLM
                if self.llm_label:
                    llm_text = f"LLM: {self.llm_decision}" if self.llm_decision else "LLM: -"
                    self.llm_label.config(text=llm_text)
                
            except Exception as e:
                print(f"GUI update error: {e}")
            
            # Schedule next update
            self.root.after(100, self.update_gui)
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.root:
            self.root.destroy()
    
    def clear_status(self):
        """Clear status (not needed for tkinter)"""
        pass
    
    def update_status(self, status: str, details: str = "", llm_decision: str = None):
        """Update the current robot status"""
        self.current_status = status.upper()
        self.action_details = details
        # Only update LLM decision if provided (don't clear existing one)
        if llm_decision is not None:
            self.llm_decision = llm_decision
    
    def update_metrics(self, attempt: int, distance: float):
        """Update performance metrics"""
        self.attempt_count = attempt
        self.distance_to_goal = distance
    
    def update_joint_angles(self, joint_angles: dict):
        """Update robot joint angles"""
        self.joint_angles = joint_angles
    
    def show_joint_angles_display(self, show: bool = True):
        """Show or hide the joint angles display"""
        self.show_joint_angles = show
        if self.root and self.joint_angles_frame:
            if show:
                self.joint_angles_frame.pack(fill=tk.X, pady=5, before=self.llm_label.master)
                self.root.geometry("400x450")  # Increase window height
            else:
                self.joint_angles_frame.pack_forget()
                self.root.geometry("400x300")  # Reset window height
    
    def display_status(self):
        """Display current status (handled automatically by GUI update loop)"""
        # Status is displayed automatically through the GUI update loop
        pass
    
    def display_action_sequence(self, actions: List[str]):
        """Display a sequence of actions (not implemented in separate window)"""
        pass
    
    def display_progress_bar(self, progress: float, label: str = "Progress"):
        """Display a progress bar (not implemented in separate window)"""
        pass
    
    def close(self):
        """Close the GUI window"""
        self.running = False
        if self.root:
            try:
                self.root.quit()
            except:
                pass


# Global instance for easy access
gui_status = GUIStatusDisplay()
