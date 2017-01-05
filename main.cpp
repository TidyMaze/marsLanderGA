#pragma GCC optimize("-O3")
#pragma GCC optimize("inline")
#pragma GCC optimize("omit-frame-pointer")
#pragma GCC optimize("unroll-loops")
#undef __NO_INLINE__
#define __OPTIMIZE__ 1

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>
#include <ctime>
#include <algorithm>    // max
#include <assert.h>
#include <limits>
#include <list>
#include <stdio.h>

#define _USE_MATH_DEFINES

#include <math.h>

#define ENV_LOCAL 1
#define ENV_CG 2
#define ENV ENV_LOCAL

#if ENV == ENV_LOCAL
#include <SDL2/SDL.h>
#endif



using namespace std;

#if ENV == ENV_CG
const int MAX_TURN_MS = 100;
#endif

#if ENV == ENV_LOCAL
const int WINDOW_WIDTH = 600;
const int WINDOW_HEIGHT = 600;
#endif

const int WIDTH = 7000;
const int HEIGHT = 3000;

#if ENV == ENV_LOCAL
const int POPULATION_SIZE = 100;
const int DEPTH = 50;
const double SELECTION_FACTOR = 0.05;
const double ELITISM_FACTOR = 0.03;
const int FRAMERATE = 30;
#endif

#if ENV == ENV_CG
const int POPULATION_SIZE = 100;
const int DEPTH = 70;
const double SELECTION_FACTOR = 0.1;
const double ELITISM_FACTOR = 0.08;
#endif

const double GRAVITY_ACC = 3.711;
const int MAX_LANDING_VSPEED = 40;
const int MAX_LANDING_HSPEED = 20;
const int MAX_VSPEED = 500;
const int MIN_VSPEED = -500;
const int MAX_HSPEED = 500;
const int MIN_HSPEED = -500;

const int STEPGRID = 35;
const int nbCol = (int)ceil(WIDTH/STEPGRID);
const int nbRow = (int)ceil(HEIGHT/STEPGRID);

static const char *const TITLE = "Mars Lander simulator - Yann Rolland - 2016";

#if ENV == ENV_LOCAL
SDL_Renderer* renderer;
SDL_Event event;
#endif

double toRadians(double angle){
    return angle * (M_PI / 180);
}

double sinDeg(double deg){
    return sin(toRadians(deg));
}

double cosDeg(double deg){
    return cos(toRadians(deg));
}

class Vector;

class Coord {
public:
    Coord();
    Coord(int x, int y);
    Coord plus(const Vector &coord) const;
    bool in(const Coord &bottomLeft, const Coord &topRight) const;
    int x, y;

    Coord times(double nb);
};

class Vector {
public:
    Vector();
    Vector(double x, double y);
    Vector plus(const Vector &vector) const;
    Vector neg() const;
    double x, y;
};

class Line {
public:
    Line();
    Line(const Coord &from, const Coord &to);
    bool collides(const Line &line) const;
    bool isHorizontal() const;
    Coord mid() const;
    Coord from, to;
};

class Rocket {
public:
    Rocket();
    Rocket(const Coord &coord, const Vector &speed);
    int idChromosome;
    Coord coord, pastCoord;
    Vector speed;
    int angle, fuel, thrust;
    bool over, inTarget;
    double score;
};

void computeGridAccess();
void showInWIndow();
void initWindow();
void handleEvents();
void parseLinesFromInput();
void moveRocketOneTurn(int depthTurn, int i, int &maxDepthYet, bool &stillOneNotOver);
void mutate(int maxDepthYet, int nbElitism, int nbToCreate, int (&newChromosomes)[POPULATION_SIZE][DEPTH*2]);
void crossover(int nbElitism, int nbSelection, int nbToCreate, int (&newChromosomes)[POPULATION_SIZE][DEPTH*2]);
void elitism(int nbElitism, int (&newChromosomes)[POPULATION_SIZE][DEPTH*2]);
int moveRocketsTillFixedState(int maxDepthYet);
void createNextGeneration(int maxDepthYet, bool oneWinner);
int countLanded(bool &oneWinner);
void startTimer(chrono::time_point<chrono::system_clock> &dstart);
void printTimer(chrono::time_point<chrono::system_clock> &dstart);
int getTimerElapsedMs(chrono::time_point<chrono::system_clock> &dstart);

void evalAllRockets();

void sortRockets();

int gridDist[nbRow][nbCol];
bool gridAccess[nbRow][nbCol];

int initialSpeedX;
int initialSpeedY;
int initialX;
int initialY;
int initialFuel;
int initialAngle;
int initialPower;

Vector::Vector(): x(0), y(0) {}
Vector::Vector(double x, double y) : x(x), y(y) {}

Vector Vector::plus(const Vector &vector) const {
    return Vector(this->x + vector.x, this->y + vector.y);
}

Vector Vector::neg() const {
    return Vector(-this->x, -this->y);
}

Coord::Coord(): x(0), y(0) {}
Coord::Coord(int x, int y) : x(x), y(y) {}

Coord Coord::plus(const Vector &vector) const {
    return Coord((int) round(this->x + vector.x), (int) round(this->y + vector.y));
}

bool Coord::in(const Coord &bottomLeft, const Coord &topRight) const{
    return this->x >= bottomLeft.x
           && this->y >= bottomLeft.y
           && this->x <= topRight.x
           && this->y <= topRight.y;
}

Coord Coord::times(double nb) {
    return Coord((int) round(this->x * nb), (int) round(this->y * nb));
}

Line::Line(): from(Coord()), to(Coord()) {}

Line::Line(const Coord &from, const Coord &to): from(from), to(to) {}

bool Line::collides(const Line &line) const{
    double p0_x = this->from.x;
    double p0_y = this->from.y;
    double p1_x = this->to.x;
    double p1_y = this->to.y;
    double s1_x = p1_x - p0_x;
    double s1_y = p1_y - p0_y;
    double s2_x = line.to.x - line.from.x;
    double s2_y = line.to.y - line.from.y;
    double s = (-s1_y * (p0_x - line.from.x) + s1_x * (p0_y - line.from.y)) / (-s2_x * s1_y + s1_x * s2_y);
    double t = ( s2_x * (p0_y - line.from.y) - s2_y * (p0_x - line.from.x)) / (-s2_x * s1_y + s1_x * s2_y);
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

bool Line::isHorizontal() const {
    return this->from.y == this->to.y;
}

Coord Line::mid() const {
    return Coord((this->from.x + this->to.x) / 2, (this->from.y + this->to.y) / 2);
}


Rocket::Rocket():coord(Coord()), speed(Vector()) {
    this->angle = initialAngle;
    this->fuel = initialFuel;
    this->thrust = initialPower;
    this->inTarget = false;
    this->over = false;
}

Rocket::Rocket(const Coord &coord, const Vector &speed):coord(coord), speed(speed) {
    this->angle = initialAngle;
    this->fuel = initialFuel;
    this->thrust = initialPower;
    this->inTarget = false;
    this->over = false;
}

vector<Line> landLines;
Rocket rockets[POPULATION_SIZE];
int chromosomes[POPULATION_SIZE][DEPTH*2];
Line targetZone;
bool quit = false;
Coord source;

#if ENV == ENV_LOCAL
vector<Line> trajectories;
#endif

void createRocket(int id){
    Rocket newRocket = Rocket(Coord(initialX, initialY), Vector(initialSpeedX, initialSpeedY));
    newRocket.idChromosome = id;
    rockets[id] = newRocket;
}

int random(int min, int max){
    return rand()%(max-min)+min;
}

bool randBool(){
    return (bool) (rand() % 2);
}

void randomChromosome(int idChromosome){
    for(int d=0;d<DEPTH;d++){
        chromosomes[idChromosome][d*2] = random(0,180+1);
        chromosomes[idChromosome][d*2+1] = random(3,4+1);
    }
}

void generateRocketsAndInitialChromosomes(){
    for(int i=0;i<POPULATION_SIZE;i++){
        randomChromosome(i);
        createRocket(i);
    }
}

bool collisionGround(int id){
    Line moveLine = Line(rockets[id].pastCoord, rockets[id].coord);
    for(int numLL=0;numLL<landLines.size();numLL++){
        Line currentLL = landLines[numLL];
        if(currentLL.collides(moveLine)){
            rockets[id].inTarget = currentLL.isHorizontal();
            return true;
        }
    }
    return false;
}

Vector force(int angle, double intensity){
    return Vector(cosDeg(angle)*intensity, sinDeg(angle)*intensity);
}


bool biggestScoreComparator(Rocket &a, Rocket &b){
    return a.score > b.score;
}

int constrainInt(int value, int vmin, int vmax){
    return min(max(value, vmin), vmax);
}

double constrainDouble(double value, double vmin, double vmax){
    return min(max(value, vmin), vmax);
}

void updateRocket(int id){
    rockets[id].pastCoord = rockets[id].coord;
    Vector acc = force(rockets[id].angle, rockets[id].thrust).plus(force(-90, GRAVITY_ACC));
    Coord loc = rockets[id].coord.plus(rockets[id].speed.plus(Vector(acc.x/2, acc.y/2)));
    Vector speed = rockets[id].speed.plus(acc);
    speed.x = constrainDouble(speed.x, MIN_HSPEED, MAX_HSPEED);
    speed.y = constrainDouble(speed.y, MIN_VSPEED, MAX_VSPEED);
    rockets[id].speed = speed;
    rockets[id].coord = loc;
}

int scale(int value, int fromMin, int fromMax, int toMin, int toMax){
    return (int) round(((double)value - fromMin) * (toMax - toMin) / (fromMax - fromMin) + toMin);
}


double eval(Rocket &rocket){
    double total = 0;

    int rY = rocket.coord.y/STEPGRID;
    int rX = rocket.coord.x/STEPGRID;

    if(rY >= nbRow || rX >= nbCol || rY < 0 || rX < 0){
        rocket.over = true;
        total -= 100000;
    }

    if(rocket.over){
        if(rocket.inTarget){
            total += 100000;
        } else {
            total -= 100000;
        }
    }

    total -= abs(90 - rocket.angle) * 200;

    double absX = abs(rocket.speed.x);
    int diffVSpeed = (int) round(abs(rocket.speed.y < -MAX_LANDING_VSPEED ? rocket.speed.y + MAX_LANDING_VSPEED : 0));
    int diffHSpeed = (int) round(absX > MAX_LANDING_HSPEED ? absX - MAX_LANDING_HSPEED : 0);

    total -= diffVSpeed * diffVSpeed * 200;
    total -= diffHSpeed * diffHSpeed * 200;

    if(!rocket.inTarget) total -= gridDist[rY][rX];
    return total;
}

#if ENV == ENV_LOCAL
void drawLine(int xa, int ya, int xb, int yb){
    int scaled_xa = scale(xa, 0, WIDTH-1, 0, WINDOW_WIDTH-1);
    int scaled_ya = scale(ya, 0, HEIGHT-1, WINDOW_HEIGHT-1, 0);
    int scaled_xb = scale(xb, 0, WIDTH-1, 0, WINDOW_WIDTH-1);
    int scaled_yb = scale(yb, 0, HEIGHT-1, WINDOW_HEIGHT-1, 0);
    SDL_RenderDrawLine(renderer, scaled_xa, scaled_ya, scaled_xb, scaled_yb);
}

void drawRect(int x, int y, int w, int h){
    int scaled_x = scale(x, 0, WIDTH-1, 0, WINDOW_WIDTH-1);
    int scaled_y = scale(y+STEPGRID, 0, HEIGHT-1, WINDOW_HEIGHT-1, 0);
    int scaled_w = scale(w, 0, WIDTH, 0, WINDOW_WIDTH);
    int scaled_h = scale(h, 0, HEIGHT, 0, WINDOW_HEIGHT);
    SDL_Rect rect;
    rect.x = scaled_x;
    rect.y = scaled_y;
    rect.w = scaled_w;
    rect.h = scaled_h;
    SDL_RenderFillRect(renderer, &rect);
}
#endif

void computeDistances(){
    source = targetZone.mid().times(1.0/STEPGRID);
    source.y += 1;
    cerr << "Source : " << source.x << " " << source.y << endl;
    list<Coord> q = list<Coord>();
    int dstMax = max(nbRow, nbCol);
    for (int i = 0; i < nbRow; i++) {
        for (int j = 0; j < nbCol; j++) {
            gridDist[i][j] = dstMax;
            q.push_back(Coord(j, i));
        }
    }
    gridDist[source.y][source.x] = 0;
    cerr << "Done init gridDist" << endl;

    while (!q.empty()) {
        Coord u;
        int minDist = std::numeric_limits<int>::max();
        std::list<Coord>::iterator minIt;
        for (std::list<Coord>::iterator it=q.begin(); it != q.end(); ++it){
            Coord uTmp = *it;
            int tmpDst = gridDist[uTmp.y][uTmp.x];
            if (tmpDst < minDist) {
                minDist = tmpDst;
                u = uTmp;
                minIt = it;
            }
        }
        q.erase(minIt);
        Coord neighborsU[] = {
                Coord(u.x - 1, u.y),
                Coord(u.x + 1, u.y),
                Coord(u.x, u.y - 1),
                Coord(u.x, u.y + 1),
        };
        if (gridAccess[u.y][u.x]) {
            for (Coord v : neighborsU) {
                if (v.x >= 0 && v.y >= 0 && v.x < nbCol && v.y < nbRow) {
                    int alt = gridDist[u.y][u.x] + 1;
                    if (alt < gridDist[v.y][v.x]) {
                        gridDist[v.y][v.x] = alt;
                    }
                }
            }
        }
    }

    cerr << "Done PF" << endl;
}

void initRandomLandlines(){
    int nbPoints = 10;
    int nbLines = nbPoints-1;
    int stepH = WIDTH/nbLines;

    int h[nbPoints];

    for(int i=0;i<nbPoints;i++){
        h[i] = random(200,1500);
    }

    Coord points[nbPoints];
    Coord points2[nbPoints];

    for(int i=0;i<nbPoints;i++) {
        points[i] = Coord(stepH*i,h[i]);
    }

    points[2] = Coord(stepH*2,h[1]);

    for(int i=0;i<nbPoints;i++) {
        points2[i] = Coord(stepH*i,random(points[i].y+600, points[i].y+1000));
    }

    for(int i=0;i<nbLines;i++){
        landLines.push_back(Line(points[i], points[i+1]));
        landLines.push_back(Line(points2[i], points2[i+1]));
    };

    targetZone = landLines[2];

    initialX = (points[nbPoints-2].x + points2[nbPoints-2].x)/2;
    initialY = (points[nbPoints-2].y + points2[nbPoints-2].y)/2;
}

void computeGridAccess() {
    cerr << "nbCol : " << nbCol << ", nbRow : " << nbRow << endl;
    for(int i=0;i<nbRow;i++){
        for(int j=0;j<nbCol;j++){
            gridAccess[i][j] = true;
        }
    }

    for(Line &l : landLines){
        int offsetX = l.to.x - l.from.x;
        int offsetY = l.to.y - l.from.y;
        for(int step=0;step<=100;step++){
            int gridIndexX = (int)(l.from.x + step*offsetX/100.0)/STEPGRID;
            int gridIndexY = (int)(l.from.y + step*offsetY/100.0)/STEPGRID;
            gridAccess[gridIndexY][gridIndexX] = false;
        }
    }
    cerr << "Done grid Access" << endl;
}

int main(int argc, char** argv)
{
    chrono::time_point<chrono::system_clock> dstart, lastSeen;
    startTimer(dstart);

    srand((unsigned int)time(NULL));

#if ENV == ENV_LOCAL
    initRandomLandlines();
    initWindow();
#endif
#if ENV == ENV_CG
    parseLinesFromInput();
#endif
    computeGridAccess();
    computeDistances();

    printTimer(dstart);

    while(!quit){
        dstart = chrono::system_clock::now();
        lastSeen = chrono::system_clock::now();

#if ENV == ENV_CG
        cin >> initialX >> initialY >> initialSpeedX >> initialSpeedY >> initialFuel >> initialAngle >> initialPower; cin.ignore();
        initialAngle += 90;

        int bestAngleSoFar = 0;
        int bestThrustSoFar = 0;
#endif
#if ENV == ENV_LOCAL
        initialSpeedX = 0;
        initialSpeedY = 0;
        initialFuel = 10000;
        initialAngle = 90;
        initialPower = 4;
#endif

        generateRocketsAndInitialChromosomes();

        int maxDepthYet = 0;
        bool keep = true;
        bool oneWinner = false;
        for(int numGen=0;!quit && keep;numGen++){

#if ENV == ENV_LOCAL
            trajectories.clear();
            handleEvents();
#endif
            maxDepthYet = moveRocketsTillFixedState(maxDepthYet);
            evalAllRockets();
            sortRockets();
#if ENV == ENV_CG
            bestAngleSoFar = chromosomes[rockets[0].idChromosome][0];
            bestThrustSoFar = chromosomes[rockets[0].idChromosome][1];
#endif
            int cntLanded = countLanded(oneWinner);
            createNextGeneration(maxDepthYet, oneWinner);
#if ENV == ENV_CG
            keep = getTimerElapsedMs(dstart) < (MAX_TURN_MS*9.0/10.0);
#endif
#if ENV == ENV_LOCAL
            if(getTimerElapsedMs(lastSeen) >= 1.0/FRAMERATE*1000) {
                showInWIndow();
                startTimer(lastSeen);
                cerr << "Done gen " << numGen << " succes : " << cntLanded*100.0/POPULATION_SIZE << "%" << endl;
            }
#endif
        }
        printTimer(dstart);
#if ENV == ENV_CG
        cout << (bestAngleSoFar-90) << ' ' << bestThrustSoFar << endl;
#endif
    }

#if ENV == ENV_LOCAL
    SDL_Quit();
#endif
    return 0;
}

void sortRockets() { sort(begin(rockets), end(rockets), biggestScoreComparator); }

void evalAllRockets() { for(int i=0; i < POPULATION_SIZE; i++) rockets[i].score = eval(rockets[i]); }

int countLanded(bool &oneWinner) {
    int count= 0;
    for(int i=0; i < POPULATION_SIZE; i++){
        if(rockets[i].inTarget
           && rockets[i].over
           && abs(rockets[i].speed.x) <= MAX_LANDING_HSPEED
           && rockets[i].speed.y >= -MAX_LANDING_VSPEED){
            // found it !
            count++;
            oneWinner = true;
        }
    }
    return count;
}

void createNextGeneration(int maxDepthYet, bool oneWinner) {
    int nbElitism = (int)floor(POPULATION_SIZE * ELITISM_FACTOR);
    int nbSelection = (int)floor(POPULATION_SIZE * SELECTION_FACTOR);
    int nbToCreate = (int)floor(POPULATION_SIZE * (1-ELITISM_FACTOR));
    assert((nbElitism + nbToCreate) == POPULATION_SIZE);

    int newChromosomes[POPULATION_SIZE][DEPTH * 2];

    // Elitism
    elitism(nbElitism, newChromosomes);
    crossover(nbElitism, nbSelection, nbToCreate, newChromosomes);
    if(!oneWinner) {
        mutate(maxDepthYet, nbElitism, nbToCreate, newChromosomes);
    }

    for(int i=0;i<POPULATION_SIZE;i++){
        createRocket(i);
    }

    copy(&newChromosomes[0][0], &newChromosomes[0][0]+POPULATION_SIZE*DEPTH*2,&chromosomes[0][0]);
}

int moveRocketsTillFixedState(int maxDepthYet) {
    bool stillOneNotOver = false;
    for(int depthTurn=0;depthTurn<DEPTH;depthTurn++){
        for(int i=0;i<POPULATION_SIZE;i++){
            moveRocketOneTurn(depthTurn, i, maxDepthYet, stillOneNotOver);
        }
        if(!stillOneNotOver) break;
    }
    return maxDepthYet;
}

void elitism(int nbElitism, int(&newChromosomes)[POPULATION_SIZE][DEPTH*2]) {
    for(int i=0; i < nbElitism; i++){
        int idChro = rockets[i].idChromosome;
        for(int ig=0;ig<DEPTH*2;ig++){
            newChromosomes[i][ig] = chromosomes[idChro][ig];
        }
    }
}

void crossover(int nbElitism, int nbSelection, int nbToCreate, int (&newChromosomes)[POPULATION_SIZE][DEPTH*2]) {
    for(int i=0; i < nbToCreate; i++){
        int firstParentIndex = rockets[random(0, nbSelection)].idChromosome;
        int secondParentIndex = rockets[random(0, nbSelection)].idChromosome;
        for(int ig=0;ig<DEPTH*2;ig++){
            int fromFirst = chromosomes[firstParentIndex][ig];
            int fromSecond = chromosomes[secondParentIndex][ig];
            newChromosomes[i+nbElitism][ig] = randBool() ? fromFirst : fromSecond;
        }
    }
}

void mutate(int maxDepthYet, int nbElitism, int nbToCreate, int (&newChromosomes)[POPULATION_SIZE][DEPTH*2]) {
    for (int i = 0; i < nbToCreate; i++) {
        for (int ig = 0; ig < DEPTH * 2; ig++) {
            int oldVal = newChromosomes[i + nbElitism][ig];
            int chanceMut = ig >= maxDepthYet*2 ? 1000 : scale(ig,0,maxDepthYet*2,200,5);
            if (random(1, chanceMut +1) == 1) {
                bool plusOuMoins = randBool();
                if (ig % 2 == 0) {
                    // angle
                    int offset = random(0, 180 + 1);
                    newChromosomes[i + nbElitism][ig] = constrainInt(plusOuMoins ? oldVal + offset : oldVal - offset, 0, 180);
                } else {
                    // thrust
                    newChromosomes[i + nbElitism][ig] = constrainInt(plusOuMoins ? oldVal + 1 : oldVal - 1, 3, 4);
                }
            }
        }
    }
}

void moveRocketOneTurn(int depthTurn, int i, int &maxDepthYet, bool &stillOneNotOver) {
    if(!rockets[i].over){
        if(depthTurn>maxDepthYet) maxDepthYet = depthTurn;
        rockets[i].angle = constrainInt(constrainInt(chromosomes[i][depthTurn*2], rockets[i].angle - 15, rockets[i].angle + 15), 0, 180);
        rockets[i].thrust = constrainInt(constrainInt(chromosomes[i][depthTurn*2+1], rockets[i].thrust - 1, rockets[i].thrust + 1), 0, 4);
#if ENV == ENV_LOCAL
        Coord old = rockets[i].coord;
#endif
        updateRocket(i);
        if(!rockets[i].coord.in(Coord(0, 0), Coord(WIDTH, HEIGHT)) || collisionGround(i)){
            rockets[i].over = true;
        } else {
            stillOneNotOver = true;
        }

#if ENV == ENV_LOCAL
        trajectories.push_back(Line(old, rockets[i].coord));
#endif
    }
}

void parseLinesFromInput() {
    int surfaceN;
    cin >> surfaceN;
    cin.ignore();
    int oldX = 0, oldY = 0;
    for (int i = 0; i < surfaceN; i++) {
        int landX;
        int landY;
        cin >> landX >> landY; cin.ignore();
        Coord newPoint = Coord(landX, landY);
        if(i>0){
            Line newLine = Line(Coord(oldX, oldY), newPoint);
            if(newLine.isHorizontal()){
                targetZone = newLine;
            }
            landLines.push_back(newLine);
        }
        oldX = newPoint.x;
        oldY = newPoint.y;
    }
}

#if ENV == ENV_LOCAL
void handleEvents() {
    while(SDL_PollEvent(&event)) {
        if( event.type == SDL_QUIT ) quit = true;
    }
}

void initWindow() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0 ) {
        fprintf(stdout,"Échec de l'initialisation de la SDL (%s)\n",SDL_GetError());
        exit(EXIT_FAILURE);
    }

    /* Création de la fenêtre */
    SDL_Window* pWindow = SDL_CreateWindow(
        TITLE,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN);

    if(!pWindow) {
        fprintf(stderr,"Erreur de création de la fenêtre: %s\n",SDL_GetError());
        exit(EXIT_FAILURE);
    }

    renderer = SDL_CreateRenderer(pWindow, -1, SDL_RENDERER_ACCELERATED);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
}

void showInWIndow() {
    SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
    SDL_RenderClear(renderer);

    for (int i = 0; i < nbRow; i++) {
        for (int j = 0; j < nbCol; j++) {
            int alpha = scale(gridDist[i][j], 0, max(nbRow, nbCol), 100, 0);
            if (gridAccess[i][j]) {
                SDL_SetRenderDrawColor(renderer, 0, 255, 0, (Uint8) alpha);
            } else {
                SDL_SetRenderDrawColor(renderer, 255, 0, 0, (Uint8) alpha);
            }
            drawRect(j * STEPGRID, i * STEPGRID, STEPGRID, STEPGRID);
        }
    }

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    drawRect(source.x * STEPGRID, source.y * STEPGRID, STEPGRID, STEPGRID);

    for (Line &l : landLines) {
        if (l.isHorizontal()) {
            SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
        } else {
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        }
        drawLine(l.from.x, l.from.y, l.to.x, l.to.y);
    }

    SDL_SetRenderDrawColor(renderer, 255, 255, 0, 5);
    for (Line &l : trajectories) {
        drawLine(l.from.x, l.from.y, l.to.x, l.to.y);
    }

    SDL_RenderPresent(renderer);
}

#endif


int getTimerElapsedMs(chrono::time_point<chrono::system_clock> &dstart) {
    chrono::time_point<chrono::system_clock> dend = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = dend - dstart;
    return (int) (elapsed_seconds.count() * 1000);
}

void printTimer(chrono::time_point<chrono::system_clock> &dstart) {
    cerr << "elapsed : " << to_string(getTimerElapsedMs(dstart)) << " ms"<< endl;
}

void startTimer(chrono::time_point<chrono::system_clock> &dstart) {
    dstart = chrono::system_clock::now();
}