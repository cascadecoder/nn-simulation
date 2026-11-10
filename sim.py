import sys, pygame, math, time, numpy as np, random, os, time, sys, json, copy

# initialize pygame stuff
pygame.init()
screenrect = pygame.Rect(0,0,800,600)
window = pygame.display.set_mode((screenrect.w,screenrect.h))
draw = pygame.Surface((screenrect.w,screenrect.h), pygame.SRCALPHA)

pygame.display.set_caption("Neural Network Sim v0.1.3")
clock = pygame.time.Clock()

# constants
mutatestrength = 0.4
mutateprob = 1/4
mutatestrengthbig = 5
mutateprobbig = 1/20
maxraylength = 80
rayamt = 3
raydih = 45
carryingcapacity = 150

# objects in simulation
objects = []
animals = []
spawners = []
buttons = []

# network size
nia = rayamt
nha = 2
noa = 2
nl = 3 # total layers
# layout of net
n = {0:nia,1:nha,2:noa}

# gdata will be current node data (first sumation, then divide by total #)
gdata = {}
for i in range(nl):
    gdata[i] = []
    for j in range(n[i]):
        gdata[i].append(0)

gdatacounts = copy.deepcopy(gdata) # will hold number of nodes in each one to divide by(will be # of animals)
gdatamax = copy.deepcopy(gdata)

gdatacollection = [] # will contain tons of gdata instances over time



fastMode = False
simStep = False
 
objectcols = {
    "food": "green",
    "danger": "red"
}

objectids = {
    "food": 50,
    "danger": -50
}

objecttypes = ["food", "danger"]
colortypes = ["darkgray", "darkblue", "darkred" , "black", "darkgreen", "yellow"]

doRerollObjects = True
chosenObject = None

# ui stuff
selected = -1
showNetwork = False

def sig(z): # sigmoid function
    if (z > 100):
        return 1
    if (z < -100):
        return 0
    return 1 / (1 + math.pow(math.e, -z))

class Object:
    def __init__(self, type, pos, dim, move):
        self.type = type
        self.pos = pos
        self.dim = dim
        self.dir = random.random() * 360
        self.rect = pygame.Rect(pos, dim)
        self.move=move
        self.dx = 0
        self.dy = 0
        objects.append(self)
        self.index = objects.index(self)
        self.chosen = False

    def render(self):
        pygame.draw.rect(window, objectcols[self.type], (self.pos, self.dim))

    def tick(self):
        global chosenObject, objects
        self.index = objects.index(self)
        if simPaused:
            if chosenObject == self:
                if (mousedown):
                    self.pos = (mousepos[0] - self.dx, mousepos[1] - self.dy)

                else:
                    chosenObject = None
            elif chosenObject == None:
                if (mousedown and mousepos[0]>=self.pos[0] and mousepos[0]<=self.pos[0]+self.dim[0] and mousepos[1]>=self.pos[1] and mousepos[1]<=self.pos[1]+self.dim[1]):
                    chosenObject = self
                    self.dx = mousepos[0]-self.pos[0]
                    self.dy = mousepos[1]-self.pos[1]
        else:
            chosenObject = None

        if (chosenObject == self):
            self.chosen = True
        else:
            self.chosen = False
        objects[self.index] = self
        self.rect = pygame.Rect(self.pos, self.dim)
        if not self.move or simPaused or True:
            return
        self.pos = ((self.pos[0] + 0.1*math.cos(math.radians(self.dir))), (self.pos[1] + 0.1*math.sin(math.radians(self.dir))))
        self.rect = pygame.Rect(self.pos, self.dim)

        if (not screenrect.colliderect(self.rect)):
            self.dir += 90
    
    def to_dict(self):
        return {"x":self.pos[0],"y":self.pos[1],"w":self.dim[0],"h":self.dim[1],"type":self.type}

class Spawner:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.pos = (x,y)
        self.w = w
        self.h = h
        self.dim = (w,h)
        self.dx  = 0
        self.dy = 0
        self.center = (x+w/2, y+h/2)

        spawners.append(self)
        self.index = spawners.index(self)
    
    def update(self):
        global chosenObject, simPaused
        if simPaused:
            if chosenObject == self:
                if (mousedown):
                    self.pos = (mousepos[0] - self.dx, mousepos[1] - self.dy)
                else:
                    chosenObject = None
            elif chosenObject == None:
                if (mousedown and mousepos[0]>=self.pos[0] and mousepos[0]<=self.pos[0]+self.dim[0] and mousepos[1]>=self.pos[1] and mousepos[1]<=self.pos[1]+self.dim[1]):
                    chosenObject = self
                    self.dx = mousepos[0]-self.pos[0]
                    self.dy = mousepos[1]-self.pos[1]
        else:
            chosenObject = None
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.center = (self.x+self.w/2,self.y+self.h/2)
        pygame.draw.rect(window, "blue", ((self.x,self.y),(self.w,self.h)))
        spawners[self.index] = self

    def to_dict(self):
        return {"x":self.x,"y":self.y,"w":self.w,"h":self.h}



class Node: # NN node
    def __init__(self, inputs, x, y, setvalue = None, inherit = False, inheritw = [], inheritb = 0, inheriti = []):
        self.inputamts = 1
        self.weights = []
        self.inputs = []
        self.results = []
        self.setvalue = None
        self.bias = 0.5
        self.output = 0
        self.total = 0
        # data for graphing (position within network)
        self.x = x
        self.y = y

        
        
        if (inherit):
            if not inputs:
                self.setvalue = setvalue
                self.weights = inheritw
                self.bias = inheritb
            else:
                self.inputamts = len(inheriti)
                self.weights = inheritw
                self.inputs = inheriti
                self.bias = inheritb

            self.shuffleWeights()
            if (random.random() < mutateprob):
                self.bias += (random.random()-0.5)*mutatestrength
            return
        #random.seed(random.random()*100000)
        if not inputs:
            self.setvalue = setvalue
            self.weights.append(random.random()*10-5)
            self.bias = random.random() * 10 * 0
        else:
            self.inputamts = len(inputs)
            self.initweights()

        

    def initweights(self):
        for i in range(self.inputamts):
            r = random.random()
            self.weights.append((r*10-5))# / self.inputamts) why is this not workgin
            random.seed(r * 1000)
        self.bias = random.random() * 10 # * self.inputamts
        
    def outputNode(self):
        
        self.results = []
        if not (self.setvalue == None):
            self.output = sig(self.setvalue*self.weights[0] - self.bias)
            return self.output
        total = 0
        for i in range(len(self.inputs)):
            self.results.append(round(self.inputs[i].output * self.weights[i] * 1000)/1000)
            total += self.inputs[i].output * self.weights[i]
        self.total = total
        self.output = sig(total/self.inputamts - self.bias)
        return self.output

    def shuffleWeights(self):
        for i in range(len(self.weights)):
            if (random.random() < mutateprob):
                r = random.random()
                self.weights[i] += (r - 0.5)*mutatestrength
                random.seed(r * 1000)
            if (random.random() < mutatestrengthbig):
                r = random.random()
                self.weights[i] += (random.random() - 0.5)*mutatestrengthbig
                random.seed(r * 1000)

    def recordData(self):
        global gdata, gdatacounts
        gdata[self.x][self.y] += self.weights[0] # only using first weights for now
        gdatacounts[self.x][self.y] += 1
        pass
 
class Ray:
    def __init__(self, p1, dir):
        self.update(p1, dir)
        self.seen = -1

    def render(self):
        if self.seen == -1:
            pygame.draw.line(window,"lightgray", self.p1, self.p2, 1)
        else:
            pygame.draw.line(window,objectcols[self.seen], self.p1, self.p2, 1)

    def update(self, p1, dir):
        self.p1 = p1
        self.dir = dir
        self.p2 = tuple(map(sum, zip(p1, (math.cos(math.radians(dir))*maxraylength,math.sin(math.radians(dir))*maxraylength))))

    def output(self):
        mindist = maxraylength * 2
        mintype = None
        for o in objects:
            res = o.rect.clipline(self.p1, self.p2)
            if (len(res) == 0):
                continue # no collision
            else:
                p3 = res[0]
                dist = math.sqrt( math.pow(p3[0]-self.p1[0] , 2) +  math.pow(p3[1]-self.p1[1] , 2))
                if (dist < mindist):
                    mindist = dist
                    mintype = o.type
        if (mintype != None):
            self.seen = mintype
        else:
            self.seen = -1
        return [mintype, mindist]
    
class Network:
    def __init__(self, parent = False):
        self.inputamt = nia #* 2
        self.timealive = 0
        self.inputs = []
        # for all the rays
        self.hiddenamt = nha
        self.hidden = []
        self.outputamt = noa
        self.outputs = []
        self.results = []
        # moveforward, turn right, turn left
        if (not parent):
            self.initNodes()
        else:
            self.inheritNodes(parent.net)
    
    def initNodes(self):
        for i in range(self.inputamt):
            self.inputs.append(Node(False, 0, i, 0)) # False, x, y, 0
        for i in range(self.hiddenamt):
            self.hidden.append(Node(self.inputs, 1, i))
        for i in range(self.outputamt):
            self.outputs.append(Node(self.hidden, 2, i))

    def inheritNodes(self, parent):
        for i in range(self.inputamt):
            self.inputs.append(Node(False, 0, i, 0, True, parent.inputs[i].weights, parent.inputs[i].bias, parent.inputs[i].inputs))
        for i in range(self.hiddenamt):
            self.hidden.append(Node(self.inputs, 1, i, 0, True, parent.hidden[i].weights, parent.hidden[i].bias, parent.hidden[i].inputs))
        for i in range(self.outputamt):
            self.outputs.append(Node(self.hidden, 2, i, 0, True, parent.outputs[i].weights, parent.outputs[i].bias, parent.outputs[i].inputs))

    def totalOutput(self):
        # Run through each layer of the network
        results = []
        for n in self.inputs:
            n.outputNode()   
            n.recordData()
        for n in self.hidden:
            n.inputs = self.inputs
            n.outputNode()
            n.recordData()
        for n in self.outputs:
            n.inputs = self.hidden
            n.outputNode()
            n.recordData()
            results.append(n.output)
        return results

    def updateInputs(self, inp):
        for i in range(len(inp)):
            self.inputs[i].setvalue = inp[i]
    
    def setNodes(self, type, arr):
        for i in range(len(arr)):
            d = arr[i] # {weights:[], bias:0}
            if (type == "inputs"):
                self.inputs[i].weights = d["w"]
                self.inputs[i].bias = d["b"]
            elif (type == "hidden"):
                self.hidden[i].weights = d["w"]
                self.hidden[i].bias = d["b"]
            else:
                self.outputs[i].weights = d["w"]
                self.outputs[i].bias = d["b"]

    def createJson(self):
        inputArray = []
        hiddenArray = []
        outputArray = []
        for input in self.inputs:
            s = {"w":input.weights,"b":input.bias}
            inputArray.append(s)
        for hidden in self.hidden:
            s = {"w":hidden.weights,"b":hidden.bias}
            hiddenArray.append(s)
        for output in self.outputs:
            s = {"w":output.weights,"b":output.bias}
            outputArray.append(s)
        return {'i': inputArray, 'h': hiddenArray, 'o': outputArray}

class Animal2:
    def __init__(self, x, y, dir, inherit = False, parent = False):
        self.child = 0
        self.hurt = 0
        self.reptime = 0
        self.timealive = 0
        self.full = 0
        self.x = x
        self.y = y
        self.dir = dir
        self.dp = 0
        self.dd = 0.1
        self.rays = []
        self.visionoutputs = []
        self.selected = False
        self.trueSelected = False
        self.readvision = []
        self.prevPos = (x,y)
        self.posses = []
        self.initVision()
        if (not inherit):
            self.net = Network()
        else:
            self.net = Network(parent)
            self.child = parent.child + 1

        animals.append(self)

    def initVision(self):
        start = self.dir-raydih/2
        if (rayamt > 1):
            dih = raydih/(rayamt - 1)
        else:
            self.rays.append(Ray((self.x, self.y), self.dir))
            return
        for i in range(rayamt):
                self.rays.append(Ray((self.x,self.y), start + i * dih))

    def updateVision(self):
        start = self.dir-raydih/2
        if (rayamt > 1):
            dih = raydih/(rayamt - 1)
        else:
            self.rays[0].update((self.x, self.y), self.dir)
            return
        i = 0
        for r in self.rays:
            r.update((self.x,self.y), start + i * dih)
            i+=1

    def update(self):
        # input from world
        self.visionOutput()
        # update NN
        #self.net.updateInputs(self.visionoutputs)
        if simPaused and not simStep:
            return
        results = self.net.totalOutput()
        
        # get output and move from there
        self.dd = (results[0] - results[1]) * 5
        self.dp = 0.4 #results[0] * 6
        if (self.visionoutputs[0] == 0 and self.visionoutputs[1] == 0 and self.visionoutputs[2] == 0):
            self.dp = 0.8
        self.dir += self.dd
        self.x += self.dp * math.cos(math.radians(self.dir))
        self.y += self.dp * math.sin(math.radians(self.dir))
        self.updateVision()

        self.handleObjects()

        self.reproduce()
    
    def visionOutput(self):
        self.visionoutputs = []
        self.readvision = []
        i = 0
        for v in self.rays:
            o = v.output()
            out = 0
            if (o[0] != None):
                #self.visionoutputs.append(objectids[o[0]]) # object type conv to #
                #self.visionoutputs.append(maxraylength-o[1]) # distance to object
                #self.readvision.append(objectids[o[0]])
                out = objectids[o[0]] * (maxraylength-o[1])
                self.visionoutputs.append(out)
            else:
                self.visionoutputs.append(0) # should i invert it, maxdist - dist
                #self.readvision.append(0)
                #self.visionoutputs.append(0)
            self.net.inputs[i].setvalue = out
            i += 1
    
    def drawRays(self):
        for r in self.rays:
            r.render()
    def render(self):
        if fastMode:
            return
        self.drawRays()
        #for r in self.rays:
        #    r.render()
        c = colortypes[self.child % (len(colortypes))]
        if self.selected:
            pygame.draw.circle(window, "black", (self.x,self.y), 8)
        pygame.draw.circle(window, c, (self.x, self.y), 5)
        pygame.draw.line(window, "black", (self.x, self.y), (self.x+math.cos(math.radians(self.dir))*10, self.y+math.sin(math.radians(self.dir))*10))

        global selected
        if True or simPaused:
            if (math.dist(mousepos,(self.x,self.y)) <= 5):
                if mousedown:
                    selected=self
                    self.selected = True
            else:
                self.selected = False
        

    def handleObjects(self):
        reached = False
        self.posses.append((self.x, self.y))
        if (len(self.posses) > 600/self.dp):
            reached = True
            self.posses.remove(self.posses[0])

        self.hurt -= 0.02 * random.random()
        self.timealive += 0.03
        self.reptime += 0.02 + 0.02 * random.random()
        if (abs(self.dp) < 0.15):
            self.hurt -= 0 * random.random()
        if (abs(self.dd) > 1 and self.visionoutputs[0] == 0 and self.visionoutputs[2] == 0 and self.visionoutputs[1] == 0):
            self.hurt -= 0.5 * random.random()
        self.visionOutput() 
        # yu981759817598715u jvkhajkgyuioahfjklashtjy3uityhjkdahgliuahtuioahlgjknajlkghajklhjahfkhaslfkhaslkfjhaeoiuyuiohbjklhjlh   make it to where sleeping makes you die
        if (self.visionoutputs[0] < 0):
            #self.hurt += self.visionoutputs[0] / 800000 * random.random()
            pass
        else:
            #self.hurt += self.visionoutputs[0] / 80000 * random.random()
            pass

        # find dist to prev pos
        if(reached):
            p1 = self.posses[0]
            p2 = (self.x, self.y)
            dist = math.sqrt( math.pow(p1[0] - p2[0],2) + math.pow(p1[1]-p2[1],2))
            if (dist < 50):
                pass
                #self.hurt += (dist - 50) / 5000000000 * random.random()

        for o in objects:
            if (o.rect.collidepoint((self.x, self.y))):
                if (o.type == "food"):
                    self.hurt += 0.5
                    #self.full += 0.1
                elif (o.type == "danger"):
                    self.hurt -= 500

    def reproduce(self):
        global recentSplit
        if (self.reptime > 30 + random.random()*30 and len(animals) < carryingcapacity):
            self.reptime = 0
            Animal2(self.x, self.y, self.dir, True, self)
            splitters.append(self)
            recentSplit = self
    def to_dict(self):
        return(self.net.createJson())

mousepos = (screenrect.w/2,screenrect.h/2)

font = pygame.font.SysFont("monospace", 15)

splitters = []

simPaused = False

class Button:
    def __init__(self, x, y, w, h, text, c, v):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.c = c
        self.pressed = False
        if not v:
            self.visible = False
        else:
            self.visible = True

        if self.text == "pause":
            self.paused = False
        if self.text == "fast":
            self.fast = False
        self.step = 0
        buttons.append(self)

    def draw(self):
        global simStep
        if (self.text == "step" and self.step > 0):
            self.step -= 1
            if (self.step <= 0):
                simStep = False
        if not self.text == "pause" and not simPaused:
            return
        pygame.draw.rect(window, self.c, pygame.Rect(self.x,self.y,self.w,self.h))
        text = font.render(str(self.text), True, "black")
        textrect = text.get_rect()
        textrect.center = (self.x+self.w/2, self.y+self.h/2)
        window.blit(text, textrect)

        # graphing == no button presses
        if (graphing):
            return
        if mousedown and mousepos[0] >= self.x and mousepos[0] <= self.x + self.w and mousepos[1] >= self.y and mousepos[1] <= self.y+self.h:
            if not self.pressed:
                self.instant()
            self.pressed = True
        else:
            self.pressed = False
    
    def instant(self):
        global simPaused, fastMode, doRerollObjects, simStep, animals
        if self.text == "pause":
            if self.paused:
                self.paused = False
                simPaused = False
            else:
                self.paused = True
                simPaused = True
        elif self.text == "fast":
            if (self.fast):
                self.fast = False
                fastMode = False
            else:
                self.fast = True
                fastMode = True
        elif self.text == "reroll":
            rerollObjects()
        elif self.text == "reset":
            if simPaused:
                resetAnimals()
        elif self.text == "spawner":
            Spawner(screenrect.w/2-10,screenrect.h/2-10,20,20)
        elif self.text == "step":
            self.step = 1
            simStep = True
        elif self.text == "respawn":
            respawn()
        elif self.text == "savemap":
            saveMap()
        elif self.text == "loadmap":
            loadMap()


selectedData = None
def respawn():
    global animals, recentSplit
    if (len(splitters) > 5):
        recentSplit = splitters[0]
        animals = []


def randomSpawn():
    index = 0
    if len(spawners) > 1: 
        index = random.randint(0,len(spawners)-1)
    
    return spawners[index].center

def resetAnimals():
    global animals
    animals = []
    for i in range(carryingcapacity):
        p = randomSpawn()
        Animal2(p[0],p[1],random.randint(0,360))

def updateJSON():
    with open('simdata.json', 'w') as f:
        anarr = []
        for a in animals:
            d=(a.to_dict())
            anarr.append(d)
        
        obarr = []
        for o in objects:
            d=(o.to_dict())
            obarr.append(d)
        saves = [{"reroll":doRerollObjects}]
        sarr = []
        for s in spawners:
            d=(s.to_dict())
            sarr.append(d)
        
        sparr = []
        for s in splitters:
            d=(s.to_dict())
            sparr.append(d)
        
        dataarr = [saves,obarr,anarr,sarr,sparr]
        f.write(json.dumps(dataarr))

def saveMap():
    global totalmap
    totalmap = []
    with open('maps.json', 'r') as f:
        d = json.load(f)
        print(d)
        if (len(d) > 0):
            totalmap = d
        else:
            totalmap = []
    with open('maps.json', 'w') as f:
        obarr = []
        for o in objects:
            d=(o.to_dict())
            obarr.append(d)
        sarr = []
        for s in spawners:
            d=(s.to_dict())
            sarr.append(d)
        datarr = [obarr, sarr]
        print("done loading arrs")
        totalmap.append(datarr)
        #d = json.load(f)
        f.write(json.dumps(totalmap))

def loadMap():
    with open('maps.json', 'r') as f:
        d = json.load(f)
        if (len(d) == 0):
            return
        else:
            l = len(d)-1
            print(len(d))
            id = input("Enter # from 0 to " + str(l) + ":")
            if (int(id) >= 0 and int(id) < len(d)):
                m = d[int(id)]
                # parse data to get the map squares and spawner locations
                odata = m[0]
                sdata = m[1]

                objects.clear() #{"x": 0, "y": 0, "w": 800, "h": 10, "type": "danger"}
                for o in odata:
                    Object(o["type"], (o["x"], o["y"]), (o["w"], o["h"]), 0)
                spawners.clear()
                for s in sdata:
                    Spawner(s["x"], s["y"], s["w"], s["h"])


def loadJSON():
    global doRerollObjects
    with open('simdata.json','r') as f:
        d = json.load(f)
        objects.clear()
        animals.clear()
        savedata = d[0]
        doRerollObjects = savedata[0]["reroll"]
        savedobjects = d[1]
        for o in savedobjects:
            pos = (o["x"],o["y"])
            dim = (o["w"],o["h"])
            type = o["type"]
            Object(type, pos, dim, False)
        if len(d) < 4:
            Spawner(screenrect.w/2-10,screenrect.h/2-10,20,20)
        else:   
            savedspawners = d[3]
            for s in savedspawners:
                Spawner(s["x"],s["y"],s["w"],s["h"])
        savedanimals = d[2]
        for a in savedanimals:
            p = randomSpawn()
            if len(a["i"]) != rayamt:
                print(str(len(a["i"])) + " does not match " + str(rayamt))
                return
            if (len(a["h"]) != Network().hiddenamt):
                print("hidden mismatch")
                return
            if (len(a["o"]) != Network().outputamt):
                print("output mismatch")
                return
            b = Animal2(p[0],p[1],random.randint(0,360))
            b.net.setNodes("inputs", a["i"])
            b.net.setNodes("hidden", a["h"])
            b.net.setNodes("outputs", a["o"])
        
        if len(animals) == 0:
            resetAnimals()
        split = d[4]
        for a in split:
            p = randomSpawn()
            if len(a["i"]) != rayamt:
                print(str(len(a["i"])) + " does not match " + str(rayamt))
                return
            if (len(a["h"]) != Network().hiddenamt):
                print("hidden mismatch")
                return
            if (len(a["o"]) != Network().outputamt):
                print("output mismatch")
                return
            b = Animal2(p[0],p[1],random.randint(0,360))
            b.net.setNodes("inputs", a["i"])
            b.net.setNodes("hidden", a["h"])
            b.net.setNodes("outputs", a["o"])
            splitters.append(b)
    
        

def clearJSON():
    with open('simdata.json', 'w') as f:
        f.write(json.dumps(""))

def rerollObjects():
    objects.clear()
    Object("danger", (0,0), (screenrect.w, 10), False)
    Object("danger", (0,screenrect.h-10), (screenrect.w, 10), False)
    Object("danger", (0,0), (10, screenrect.h), False)
    Object("danger", (screenrect.w-10,0), (10, screenrect.w), False)
    for i in range(40):
        Object(objecttypes[round(random.random())], (random.random()*screenrect.w, random.random()*screenrect.h), (random.random()*50+25, random.random()*50+25), True)
#random objects objecttypes[round(random.random())]
if doRerollObjects:
    rerollObjects()
# check if file has data
f = open("simdata.json")

j = f.read()
if len(j) > 0:
    loadJSON()
else:
    Spawner(250,250,20,20)
    resetAnimals()

def renderObjects():
    global chosenObject
    for i in objects:
        i.tick()
        i.render()
    if (deleting):
        for i in objects:
            if chosenObject == i:
                objects.remove(i)
                del i
                chosenObject = None
                return
def renderSpawners():
    for i in spawners:
        i.update()
def renderButtons():
    for i in buttons:
        i.draw()

bestanimal = None
recentSplit = None
ticker = 0

def txt(s,pos):
    text = font.render(str(s), True, "black")
    textrect = text.get_rect()
    textrect.bottomleft = pos
    window.blit(text, textrect)

def displaySelected():
    if simPaused:
        txt("Paused", (160,50))
    if selected != -1:
        d=(selected.to_dict())
        ca = selected
        ca.drawRays()
        txt("Animal Stats", (160,80))
        txt("Speed: " + str(ca.dp), (160, 100))
        txt("dDir: " + str(ca.dd), (160,120))
        txt("Vision: " + str(ca.visionoutputs), (160,140))

def renderAnimals():
    
    global ticker
    global selected
    if (selected not in animals):
        selected = -1
    if not simPaused or simStep:
        ticker += 1
    choppingblock = []

    # reset graph data here
    for i in range(nl):
        for j in range(n[i]):
            gdata[i][j] = 0
            gdatacounts[i][j] = 0
            gdatamax[i][j] = 0
    
    for i in animals:
        #if not simPaused:
        i.update()
        i.render()

        # kill bad animals
        if (not simPaused or simStep) and (i.hurt < -10 or i.timealive > 200 or i.full > 50) and len(animals) > 1:
            choppingblock.append(i)

    for i in range(nl): # calculate average
        for j in range(n[i]):
            if (gdatacounts[i][j] > 0):
                gdata[i][j] = gdata[i][j] / gdatacounts[i][j]
    
    if (ticker % 10 == 0 and not simPaused):
        gdatacollection.append(copy.deepcopy(gdata)) # record dataset
        
    
    for i in reversed(choppingblock):
        if i == selected:
            selected = -1
        animals.remove(i)

    while (len(splitters) > 10):
        splitters.remove(splitters[0])

    #if (mousedown and len(animals) < carryingcapacity):
    #    Animal2(mousepos[0],mousepos[1],random.randint(0,360), True, bestanimal)
    if (not simPaused and ticker % 5000 == 0):
        respawn()
    if (len(animals) == 0):
        if recentSplit is not None:
            p = randomSpawn()
            Animal2(p[0],p[1],random.randint(0,360), True, recentSplit)
    if (len(animals) < carryingcapacity / 4 and recentSplit is not None): # maybe instead of just doing the top use the one that most recently split
        for i in range(round((carryingcapacity-len(animals))/4)):
            p = randomSpawn()
            Animal2(p[0],p[1],random.randint(0,360), True, splitters[i % len(splitters)])

    # update json file
    #f = open("simdata.json", "w")
    #i = json.dump()
    #h = json.dump()
    #o = json.dump()

    if (ticker % 400 == 0):
        updateJSON()
        
    
    

    if True:
        displaySelected()
    

b = Button(50,50,100,50,"pause","lightblue",True)
b2 = Button(50, 400, 100, 50, "fast","teal",True)
b3 = Button(50, 500, 100, 50, "reroll", "yellow",True)
b4 = Button(500, 50, 100, 50, "reset", "darkred", True)
b5 = Button(500, 400, 100, 50, "spawner", "darkblue", True)
b6 = Button(600, 500, 100, 50, "step", "orange", True)
b7 = Button(600, 200, 100, 50, "respawn", "pink", True)
b8 = Button(400,500,100,50, "savemap", "lightgreen", True)
b9 = Button(290,500,100,50, "loadmap", "lightpink", True)

# Main Sim Loop
running = True
mousedown = False
edit = False
placingshape = False
deleting = False

# graphing variables
graphing = False
graphMode = "selection"
graphSelectionInit = True
graphSelectedNode = -1
gbtns = []
class GraphButton:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.smooth = 0
        self.smooth2 = 0
        self.id = len(gbtns)
        gbtns.append(self)
    
    def tick(self):
        global graphSelectedNode
        a = 200 + self.smooth * 30
        self.smooth2 += (1-self.smooth2) / 10
        if (graphSelectedNode < 0 and math.sqrt( math.pow(mousepos[0] - self.x,2) + math.pow(mousepos[1] - self.y, 2)) <= self.r):
            self.smooth += (1 - self.smooth) / 15
            pygame.draw.circle(draw, pygame.Color(255,255,255,230), (self.x, self.y), self.r*self.smooth2 + self.smooth * 4)

            if (mousedown):
                if (graphSelectedNode < 0):
                    graphSelectedNode = self.id
                
        else:
            self.smooth += (0 - self.smooth) / 4
            pygame.draw.circle(draw, pygame.Color(255,255,255,200), (self.x, self.y), self.r*self.smooth2 + self.smooth * 4)

def createRows(amt, height, margin):
    if (amt == 1):
        return [height / 2]
    h = height - margin * 2
    d = h / (amt + 1)
    r = []
    for i in range(amt):
        r.append(margin + d * (i+1))
        
    return r


newshape = {"x":0,"y":0,"w":0,"h":0,"type":0}
while (running):
    # User events
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            running = False
        if (event.type == pygame.MOUSEMOTION):
            mousepos = pygame.mouse.get_pos()
        if (event.type == pygame.MOUSEBUTTONDOWN):
            mousedown = True
        if (event.type == pygame.MOUSEBUTTONUP):
            mousedown = False
        if (event.type == pygame.KEYDOWN):
            if (event.dict["unicode"] == 'e'):
                edit = True
            elif (event.dict["unicode"] == 'q'):
                edit = False
            if (edit and event.dict["unicode"] == ' '):
                newshape['type'] = 1 - newshape['type']
            if (event.dict["unicode"] == 'x'):
                deleting = True
            if (event.dict["unicode"] == 'g'):
                graphing = True
            if (event.dict["unicode"] == 'h'):
                if (graphing):
                    graphSelectedNode = -1
                    graphMode = "selection"
        if (event.type == pygame.KEYUP):
            if (event.dict["unicode"] == 'x'):
                deleting = False
        
    window.fill("white")

    
    if not fastMode:
        renderObjects()
        renderSpawners()
    renderAnimals()
    renderButtons()

    #doRerollObjects = False

    if edit:
        if mousedown:
            
            if (placingshape):
                newshape["w"] = mousepos[0]-newshape["x"]
                newshape["h"] = mousepos[1]-newshape["y"]
            else:
                placingshape = True
                newshape["x"] = mousepos[0]
                newshape["y"] = mousepos[1]
            pygame.draw.rect(window, objectcols[objecttypes[newshape["type"]]], ((newshape["x"], newshape["y"]), (newshape["w"],newshape["h"])))
        else:
            if (placingshape):
                edit = False
                Object(objecttypes[newshape["type"]], (newshape["x"], newshape["y"]), (newshape["w"],newshape["h"]),False)
            placingshape = False
    else:
        placingshape = False
    

    clock.tick()
    fps = clock.get_fps()
    txt(round(fps), (10,600))

    if (graphing):
        pygame.draw.rect(draw, pygame.Color(50,60,140,240), screenrect)
        marginx = 125
        marginy = 75
        locations = [marginx, screenrect.w/2, screenrect.w - marginx]
        texts = ["input", "hidden", "output"]
        if (graphMode == "selection"):
            # display neural network
            if (graphSelectionInit):
                graphSelectionInit = False
                ys = [nia, nha, noa]
                for a in range(len(ys)):
                    x = locations[a]
                    y = ys[a]
                    s = createRows(y, screenrect.h, marginy)
                    for i in s:
                        GraphButton(x, i, 25)
            else:
                for b in gbtns:
                    b.tick()
                for x in range(len(locations)):
                    
                    text = font.render(str(texts[x]), True, "white")
                    textrect = text.get_rect()
                    textrect.bottomleft = (locations[x]-textrect.w/2, 100)
                    draw.blit(text, textrect)

                if (graphSelectedNode >= 0):
                    graphMode = "graph"
        elif graphMode == "graph":
            # display current graph of node
            # good luck bro 
            # thanks i need it
            dist = screenrect.w - marginx * 2
            pygame.draw.line(draw, pygame.Color(255,255,255,240), (marginx, screenrect.h / 2), (screenrect.w - marginx, screenrect.h / 2))
            

            

            # pos of current node
            gx = 0
            gy = 0
            a = graphSelectedNode
            for i in range(len(n)):
                a -= n[i]
                if (a >= 0):
                    #print(str(a) + "nope" + str(n[i]))
                    gx += 1
                else:
                    break
            gy = n[gx] + a
            gmax = 0
            # max of current node
            for g in gdatacollection:
                if (abs(gdatamax[gx][gy]) < abs(g[gx][gy])):
                    gdatamax[gx][gy] = g[gx][gy]
                    gmax = g[gx][gy]
           
            # DRAW THE EFFING THING
            if (len (gdatacollection) > 400):
                domainmin = len(gdatacollection) - 400
            else:
                domainmin = 0

            dx = 0
            if (len(gdatacollection) > 1):
                dx = dist / (len(gdatacollection)-1 - domainmin)
            
            for bx in range(nl):
                for by in range(n[bx]):
                    prev = (0,0)
                    i = domainmin
                    ir = 0
                    if (len(gdatacollection) < domainmin):
                        break
                    
                    for index in range(len(gdatacollection)-domainmin):
                        g= gdatacollection[index+domainmin]
                        drawx = marginx + (ir)*dx
                        drawy = (-1) * (g[bx][by] / gmax * (screenrect.h/2-marginy)) + screenrect.h / 2 # 100 should be gmax
                        c = "gray"
                        if (bx == gx and by == gy):
                            c = "green"
                        else:
                            pass
                        pygame.draw.circle(draw, c, (drawx, drawy), 0.5)
                        if (ir > 0):
                            pygame.draw.line(draw, c, prev, (drawx, drawy))

                        if (i == len(gdatacollection) - 1):
                            text = font.render(str(bx) + ", " + str(by), True, "white")
                            textrect = text.get_rect()
                            textrect.bottomleft = (drawx + 10, drawy)
                            draw.blit(text, textrect)

                        prev = (drawx,drawy)
                        i += 1
                        ir += 1
            

    window.blit(draw, (0,0))
    pygame.display.flip()
    #clock.tick(60)
    

pygame.quit()
