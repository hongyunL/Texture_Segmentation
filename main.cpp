
/* CVUT 2016 - main file */
/* segmenter v1.4 */

#include "Segmenter.h"
#include "DyXML_wrapper.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Global.h"
#include "TBAlloc.h"
#include "Pres2.h"
#include "Maths.h"
#include "KMeans.h"


// Derive your own segmenter class
class MySegmenter : public Segmenter {
  virtual int features(DyXML_wrapper &par);
//  virtual int clustering(DyXML_wrapper &par);
//  virtual int postprocessing(DyXML_wrapper &par);
};


/**
    Performs preprocessing of image by moving window of size nr_k * nc_k
    and subtracting calculated average in that window from central value.

    @param img : input matrix
    @param nimg : number of layers in image (3 for RGB)
    @param nr : number of rows
    @param nc : number of columns
    @param nr_k : number of rows in convolution matrix
    @param nc_k : number of columns in convolution matrix
    @return averaged matrix
*/
float*** convolution_avg (float*** img, int nimg, int nr, int nc, int nr_k, int nc_k){
  int kCenterC = nc_k / 2, kCenterR = nr_k / 2, r1, c1, sum, avg;
  float*** img2;
  vr_alloc(img2,0,nimg-1,0,nr-1,0,nc-1);
  for (int r=0;r<nr;r++){
    for (int c=0;c<nc;c++) {
     for (int i=0;i<nimg;i++) {
      sum = 0;
      for (int rk=0;rk<nr_k;rk++){
        for (int ck=0;ck<nc_k;ck++) {
          r1 = r - kCenterR + rk;
          c1 = c - kCenterC + ck;
          if (r1 < nr && r1 >= 0 && c1 < nc && c1 >=0){
              sum += img[i][r1][c1];
          }
        }
       }
       avg = sum/(nr_k*nc_k);
       img2[i][r][c] = img[i][r][c] - avg;
      }
    }
  }
  return img2;
}


/**
    Performs convolution by masks on image.

    @param img : input matrix
    @param nimg : number of layers in image (3 for RGB)
    @param nr : number of rows
    @param nc : number of columns
    @param masks : array of matrices
    @param nm : number of masks
    @param nr_k : number of rows in convolution matrices
    @param nc_k : number of columns in convolution matrices
    @return array of result images: each image is result of convolution by certain matrix.
*/
float**** convolution_by_masks (float*** img, int nimg, int nr, int nc, int*** masks, int nm, int nr_k, int nc_k){
  int kCenterC = nc_k / 2, kCenterR = nr_k / 2, r1, c1, sum;
  float**** imgs;
  vr_alloc(imgs, 0, nm-1 ,0,nimg-1,0,nr-1,0,nc-1);
  for (int m=0; m<nm; m++) {
      for (int r=0;r<nr;r++){
        for (int c=0;c<nc;c++) {
         for (int i=0;i<nimg;i++) {
          sum = 0;
          for (int rk=0;rk<nr_k;rk++){
            for (int ck=0;ck<nc_k;ck++) {
              r1 = r - kCenterR + rk;
              c1 = c - kCenterC + ck;
              if (r1 < nr && r1 >= 0 && c1 < nc && c1 >=0){
                  sum += img[i][r1][c1] * masks[m][rk][ck];
              }
            }
           }
           imgs[m][i][r][c] = sum;
          }
        }
      }
    }
  return imgs;
}


/**
    Generates masks from vectors by multiplying them one by one.

    @param vectors : vectors
    @param nv : number of vectors
    @param l : length of vectors
    @return array of matrices
*/
int*** generate_masks (int** vectors, int nv, int l){
    int*** matrices;
    int counter = 0;
    vr_alloc(matrices,0,nv*nv-1,0,l-1,0,l-1);
    for(int i=0; i<nv; i++){
        for(int j=0; j<nv; j++){
          for (int r=0; r<l; r++){
              for (int c=0; c<l; c++){
                matrices[counter][r][c] = vectors[i][r] * vectors[j][c];
              }
          }
          counter++;
        }
    }
    return matrices;
}


/**
    Produces energy maps from images.

    @param imgs : array of images
    @param nimg : number of layers in image (3 for RGB)
    @param nr : number of rows in image
    @param nc : number of columns in image
    @param nm : number of masks
    @return array of energy masks
*/
float**** produce_energy_maps (float**** imgs, int nimg, int nr, int nc, int nm){
  int sum;
  float**** res;
  vr_alloc(res, 0, nm-1 ,0,nimg-1,0,nr-1,0,nc-1);
  for (int m=0; m<nm; m++) {
      for (int r=0;r<nr;r++){
        for (int c=0;c<nc;c++) {
         for (int i=0;i<nimg;i++) {
          sum = 0;
          for (int r1=r-7;r1<=r+7;r1++){
            for (int c1=c-7;c1<=c+7;c1++) {
              if (r1 < nr && r1 >= 0 && c1 < nc && c1 >=0){
                  sum += abs(imgs[m][i][r1][c1]);
              }
            }
           }
           res[m][i][r][c] = sum;
          }
        }
      }
    }
  return res;
}


/**
    Combines masks by averaging to form 9 resulting masks.

    @param imgs : input images
    @param nimg : number of layers in image (3 for RGB)
    @param nr : number of rows in image
    @param nc : number of columns in image
    @param nm : number of masks
    @param pairs : pairs of indexes of those matrices, that have to be combined by averaging
    @param np : number of pairs
    @return combined matrices
*/
float**** combine_maps(float**** imgs, int nimg, int nr, int nc, int nm, int** pairs, int np){
  float**** res;
  vr_alloc(res, 0, np-1 ,0,nimg-1,0,nr-1,0,nc-1);
  for (int p=0; p<np; p++) {
    for (int i=0;i<nimg;i++) {
      for (int r=0;r<nr;r++){
        for (int c=0;c<nc;c++) {
           res[p][i][r][c] = (imgs[pairs[p][0]][i][r][c] + imgs[pairs[p][1]][i][r][c])/2;
          }
        }
      }
    }
    return res;
}


/**
    Reshapes features in such a way, that it transforms 4D structure into 3D:
     it flatens levels responsible for number of masks and number of layers,
     so that for each pixel in image we have a 1D array of features.

    @param features : input features
    @param nm : number of masks
    @param nimg : number of layers in image (3 for RGB)
    @param nr : number of rows in image
    @param nc : number of columns in image
    @return reshaped features
*/
float*** reshape_features(float**** features, int nm, int nimg, int nr, int nc){
  float*** res;
  vr_alloc(res,0,nm*nimg-1,0,nr-1,0,nc-1);
  for (int r=0;r<nr;r++){
    for (int c=0;c<nc;c++) {
      for (int m=0; m<nm; m++) {
        for (int i=0;i<nimg;i++) {
           res[m*nimg+i][r][c] = features[m][i][r][c];
        }
       }
    }
  }
  return res;
}


/**
    Frees memory from 4D float array.

    @param a : input array
    @param nm : size of array on level 1
    @param nimg : size of array on level 2
    @param nr : size of array on level 3
    @param nc : size of array on level 4
*/
void free_4d(float**** a, int nm, int nimg, int nr, int nc){
  for (int m=0; m<nm; m++) {
    for (int i=0;i<nimg;i++) {
      for (int r=0;r<nr;r++){
        free(a[m][i][r]);
       }
       free(a[m][i]);
    }
    free(a[m]);
  }
  free(a);
}


/**
    Frees memory from 3D float array.

    @param a : input array
    @param nimg : size of array on level 1
    @param nr : size of array on level 2
    @param nc : size of array on level 3
*/
void free_3d(float*** a, int nimg, int nr, int nc){
    for (int i=0;i<nimg;i++) {
      for (int r=0;r<nr;r++){
        free(a[i][r]);
       }
       free(a[i]);
    }
    free(a);
}


/**
    Frees memory from 2D int array.

    @param a : input array
    @param nr : size of array on level 1
    @param nc : size of array on level 2
*/
void free_2d_int(int** a, int nr, int nc){
  for (int r=0;r<nr;r++){
    free(a[r]);
   }
   free(a);
}


/**
    Frees memory from 3D int array.

    @param a : input array
    @param nimg : size of array on level 1
    @param nr : size of array on level 2
    @param nc : size of array on level 3
*/
void free_3d_int(int*** a, int nimg, int nr, int nc){
  for (int i=0;i<nimg;i++) {
      for (int r=0;r<nr;r++){
        free(a[i][r]);
       }
       free(a[i]);
    }
    free(a);
}


int MySegmenter::features(DyXML_wrapper &par) {

 float sigma;
  const char *filename;
  int d,key;

// load parameters
  par.select("features");
  par.get_val_opt(sigma,"sigma",1.0f,"Sigma:");
  par.get_val_opt(d,"win",3,"Window size:");
  par.back();


// commented code below was used for calculating minimum accuracy
// by taking grey scale
/*
  nfeat=1;
  vr_alloc(feat,0,nfeat-1,0,nr-1,0,nc-1);

  for (int r=0;r<nr;r++){
    for (int c=0;c<nc;c++) {
      feat[0][r][c] = 0;
      for (int i=0;i<nimg;i++) {
        feat[0][r][c] += img[i][r][c]/3;
      }
    }
  }
*/


  // perform number of transformations for generating final array of features
  int CONVOLUTION_SIZE = 15;
  float*** img_preprocessed = convolution_avg(img, nimg, nr, nc, CONVOLUTION_SIZE, CONVOLUTION_SIZE);

  // vectors are defined in Law's algorithm
  int** vectors;
  vr_alloc(vectors,0,4,0,5);
  vectors[0][0]=1; vectors[0][1]=4; vectors[0][2]=6; vectors[0][3]=4; vectors[0][4]=1;
  vectors[1][0]=-1; vectors[1][1]=-2; vectors[1][2]=0; vectors[1][3]=2; vectors[1][4]=1;
  vectors[2][0]=-1; vectors[2][1]=0; vectors[2][2]=2; vectors[2][3]=0; vectors[2][4]=-1;
  vectors[3][0]=1; vectors[3][1]=-4; vectors[3][2]=6; vectors[3][3]=-4; vectors[3][4]=1;

  int*** masks = generate_masks(vectors, 4, 5);
  free_2d_int(vectors, 5, 6);

  float**** filtered_images = convolution_by_masks(img_preprocessed, nimg, nr, nc, masks, 16, 5, 5);
  free_3d(img_preprocessed, nimg, nr, nc);
  free_3d_int(masks, 16, 5, 5);

  float**** energy_masks = produce_energy_maps(filtered_images, nimg, nr, nc, 16);
  free_4d(filtered_images, 16, nimg, nr, nc);

  // the way pairs of masks are combined is defined in Law's algorithm
  int** pairs;
  vr_alloc(pairs,0,8,0,1);
  pairs[0][0]=1; pairs[0][1]=4; pairs[1][0]=2; pairs[1][1]=8; pairs[2][0]=3; pairs[2][1]=12;
  pairs[3][0]=5; pairs[3][1]=5; pairs[4][0]=6; pairs[4][1]=9; pairs[5][0]=7; pairs[5][1]=13;
  pairs[6][0]=10; pairs[6][1]=10; pairs[7][0]=11; pairs[7][1]=14; pairs[8][0]=15; pairs[8][1]=15;

  float**** combined_maps = combine_maps(energy_masks, nimg, nr, nc, 16, pairs, 9);
  free_4d(energy_masks, 16, nimg, nr, nc);
  free_2d_int(pairs, 9, 2);

  feat = reshape_features(combined_maps, 9, nimg, nr, nc);
  free_4d(combined_maps, 9, nimg, nr, nc);
  nfeat = nimg*9;

// load optional parameters
  par.get_str_opt(filename,"FeatFile",NULL,"Enter name of features:");
  par.get_val_opt(key,"FeatType",PRES_WRITE_KEY_PNG,"Enter features format code:");
// save features as image
  if (filename) { int ncolour=nfeat-1, nlev=255; pres2(key,feat,&nc,&nr,&ncolour,&nlev,filename); }

  return 0;
}

/*int MySegmenter::clustering(DyXML_wrapper &par) {

// load parameters
// par. ...

// allocate probability maps array and initialize
  vr_alloc(probs,0,nclust-1,0,nr-1,0,nc-1);

// compute probability maps
// ...

  return 0;
}*/

/*int MySegmenter::postprocessing(DyXML_wrapper &par) {

// obtain thematic map directly
  thematic_map(par);
// or call the base class function
  Segmenter::postprocessing(par);

// load parameters
// par. ...

// do some postprocessing
// ...

  return 0;
}*/


// Main function - maintains processing loop and parameters
int main(int argc, char* argv[]) {
  DyXML_wrapper par(argc,argv,"Segmenter");    // load parametric file

  par.init_runs();                             // initialize default parameters
  while (par.next_run()) {                     // main loop over all computation runs
    MySegmenter seg;                           // instance of the overriden class

    seg.run(par);                              // run segmentation
  }

  return 0;
}
