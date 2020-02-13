# Snapchat Filters
Basic desktop application to play around with Snapchat-alike filters like mask, sunglasses and santa hat automatic in-face superposition in real time.
**You can use multiple filters at once.**

### Currently following filters are available:
* Santa hat
* Santa beard
* Sunglasses
* Mask 
* Hat
* Mustache

<p align="center">
  <img width="420" height="245" src="https://i.imgur.com/bcMhy6o.png">
  <img width="420" height="245" src="https://i.imgur.com/iRJsu0g.png">
  <img width="420" height="245" src="https://i.imgur.com/gd4ALDU.png">
  <img width="420" height="245" src="https://i.imgur.com/0Z2xSyK.png">
  <img width="420" height="245" src="https://i.imgur.com/iKEqapx.png">
  <img width="420" height="245" src="https://i.imgur.com/Br3VM9P.png">
</p>

## Bounderies
You can also view bounderies of detected areas (faces and eyes)
<p align="center">
  <img width="420" height="245" src="https://i.imgur.com/ipMHmKi.png">
</p>

## To-do:
* Filter: rainbow spilling out from mouth (after opening it)
* Create custom images for filters in Adobe PS
* Add new filters

## Usage
Example with single filter:
```
python webcam.py -f sunglasses
```
Example with multiple filters:
```
python webcam.py -f sunglasses santahat santabeard
```
Example with detected boundries:
```
python webcam.py -f santahat -b
```
