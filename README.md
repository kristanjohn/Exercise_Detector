# Exercise Detector

Small project designed to learn and recognise different human movements.
Input collected from two sensortags in synchronises. Each sensortag uses accelerometer and gyroscope peripherals. 
Sensortag locations: left thigh and chest.

To learn new a new movement add sample data to ./data/training with the name *movement_name*

Training Data contains: 
- Pushups
- Sit-ups
- Squats
- Sitting
- Standing
- Plank
- Starjumps

## Argument -validate
Uses training data from ./data/training to class Exercises
Validates data from ./data/Validates
Results are displayed in a Confusion Matrix

## Argument -stream
Uses accelerometer and gyroscope values
These classifications and raw data are uploaded to a database
where they can be downloaded from an app
