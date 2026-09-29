from random import random

import pygame

pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(16)

background_channel = pygame.mixer.Channel(0)

screen = pygame.display.set_mode((500, 300))
pygame.display.set_caption("Musport Yoga")

clock = pygame.time.Clock()

class ContinuousSound:
    def __init__(self, sound, channel_number):
        self.sound = sound
        self.channel = pygame.mixer.Channel(channel_number)

        self.current_volume = 0.0
        self.target_volume = 0.0

    def start(self):
        self.channel.set_volume(0.0)
        self.channel.play(self.sound, loops=-1)

    def set_intensity(self, intensity):
        self.target_volume = max(0.0, min(1.0, intensity))

    def update(self):
        self.current_volume += (
            self.target_volume - self.current_volume
        ) * 0.03

        self.channel.set_volume(self.current_volume)


forest_background = pygame.mixer.Sound("sounds/forest_background.wav")
breeze = pygame.mixer.Sound("sounds/breeze.wav")
birds1 = pygame.mixer.Sound("sounds/birds1.wav")
birds2 = pygame.mixer.Sound("sounds/birds2.wav")
birds3 = pygame.mixer.Sound("sounds/birds3.wav")
cricket = pygame.mixer.Sound("sounds/cricket.wav")
heavy_rain = pygame.mixer.Sound("sounds/heavy_rain.wav")
light_rain = pygame.mixer.Sound("sounds/light_rain.wav")
long_rain = pygame.mixer.Sound("sounds/long_rain.wav")
stepping_on_leaves = pygame.mixer.Sound("sounds/stepping_on_leaves.wav")
water_flowing = pygame.mixer.Sound("sounds/water_flowing.wav")
wind_blowing = pygame.mixer.Sound("sounds/wind_blowing.wav")

water = ContinuousSound(water_flowing, 1)
wind = ContinuousSound(wind_blowing, 2)
rain = ContinuousSound(long_rain, 3)

bird_channel = pygame.mixer.Channel(4)
cricket_channel = pygame.mixer.Channel(5)
breeze_channel = pygame.mixer.Channel(6)
leaves_channel = pygame.mixer.Channel(7)
other_rain_channel = pygame.mixer.Channel(8)

forest_background.set_volume(0.2)

breeze.set_volume(0.8)
birds1.set_volume(0.8)
birds2.set_volume(0.8)
birds3.set_volume(0.8)
cricket.set_volume(0.8)
heavy_rain.set_volume(0.8)
light_rain.set_volume(0.8)
long_rain.set_volume(0.8)
stepping_on_leaves.set_volume(0.8)
water_flowing.set_volume(1.0)       
wind_blowing.set_volume(0.8)    
  


def trigger_birds():
    birds_sounds = [birds1, birds2, birds3]
    bird_channel.play(birds_sounds[int(random() * len(birds_sounds))])

def trigger_breeze():
    breeze_channel.play(breeze)

def trigger_cricket():
    cricket_channel.play(cricket)  

def trigger_leaves():
    leaves_channel.play(stepping_on_leaves)

def trigger_other_rain():
    other_rain_channel.play(light_rain)


background_channel.play(forest_background, loops=-1)

print("Musport Yoga started!")
print("Press B to trigger the breeze sound.")
print("Press 1 to trigger the birds sound.")
print("Press 2 to trigger the rain sound.")
print("Press 3 to trigger the wind sound.")
print("Press 4 to trigger the water sound.")
print("Press 5 to trigger the leaves sound.")
print("Press C to trigger the cricket sound.")
print("Press Q to quit.")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:
                trigger_breeze()
            elif event.key == pygame.K_c:
                trigger_cricket()
            elif event.key == pygame.K_1:
                trigger_birds()
            elif event.key == pygame.K_2:
                rain.start()
                rain.set_intensity(0.8)
            elif event.key == pygame.K_3:
                wind.start()
                wind.set_intensity(0.8)
            elif event.key == pygame.K_4:
                water.start()
                water.set_intensity(0.8)
            elif event.key == pygame.K_5:
                trigger_leaves()       
            elif event.key == pygame.K_q:
                pygame.quit()
                quit()
            

    water.update()

    wind.update()

    rain.update()

    



    clock.tick(60)