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
* Clown hair
* Clown nose
* Eyeballs

<p align="center">
  <img width="250" height="145" src="https://i.imgur.com/bcMhy6o.png">
  <img width="250" height="145" src="https://i.imgur.com/iRJsu0g.png">
  <img width="250" height="145" src="https://i.imgur.com/gd4ALDU.png">
  <img width="250" height="145" src="https://i.imgur.com/0Z2xSyK.png">
  <img width="250" height="145" src="https://i.imgur.com/qWn6QO9.png">
  <img width="250" height="145" src="https://i.imgur.com/Br3VM9P.png">
  <img width="250" height="145" src="https://i.imgur.com/86bLEwz.png">
  <img width="250" height="145" src="https://i.imgur.com/bNOQTIn.png">
  <img width="250" height="145" src="https://i.imgur.com/yRDQnwy.png">
</p>

You can chose package of filters with "-p" parameter. 
Currently following filter packages are available:
* Clown
* Man
* Santa

## Bounderies
You can also view bounderies of detected areas (faces and eyes)
<p align="center">
  <img width="420" height="245" src="https://i.imgur.com/ipMHmKi.png">
</p>

## To-do:
* Filter: rainbow spilling out from mouth (after opening it)
* GIF format
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
