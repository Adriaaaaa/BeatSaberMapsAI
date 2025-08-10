
import os
from analysis import BeatSaberMapAnalyzer
from generation import generator

MAPS_DIR = "Favorites"

def main():
  print("Welcome to Beat Saber Maps AI by Adria!\n")
  print("This program will analyze your Beat Saber maps and provide statistics.\n")  
  
  #Menu
  while True:
    print("\nWhat do you want to do ? :\n")
    print("1.Analyse all maps\n")
    print("2.Visualize a map\n")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == '1':
      print("Analyzing all maps...\n")
      for map_folder in os.listdir(MAPS_DIR):
        map_path = os.path.join(MAPS_DIR, map_folder)
        if os.path.isdir(map_path):      
          analyseur = BeatSaberMapAnalyzer(map_path)
          analyseur.load_data()
          print(f"Stats for {map_folder}: {analyseur.analyze_maps()}\n")           
      break
    elif choice == '2':
      print("Visualizing a map is not implemented yet.\n")
      for map_folder in os.listdir(MAPS_DIR):
        map_path = os.path.join(MAPS_DIR, map_folder)
        if os.path.isdir(map_path):      
          analyseur = BeatSaberMapAnalyzer(map_path)
          analyseur.load_data()       
          analyseur.visualize_map()  # Assuming visualize_map is implemented
          break
      continue
    else:
      print("Invalid choice, please enter 1 or 2.\n")

if __name__ == "__main__":
  main()
    