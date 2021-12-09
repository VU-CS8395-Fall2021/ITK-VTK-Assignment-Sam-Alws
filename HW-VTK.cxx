#include "itkImage.h"
#include "itkImageFileReader.h"

#include "itkImageToVTKImageFilter.h"

#include "vtkSmartPointer.h"
#include "vtkImageSliceMapper.h"
#include "vtkImageActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkCamera.h"
#include "vtkInteractorStyleImage.h"
#include "vtkCommand.h"
#include "itkMatrix.h"
#include "itkDiffusionTensor3D.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTensorFractionalAnisotropyImageFilter.h"
#include <queue>
#include "itkMultiplyImageFilter.h"
#include "vtkPolyDataMapper.h"
#include "vtkLineSource.h"
#include "vtkSphereSource.h"
#include "vtkProperty.h"
#include "vtkRendererCollection.h"
#include "vtkImageData.h"
#include "vtkVersion.h"
#include "vtkPointPicker.h"

const int nDims = 3;
typedef itk::DiffusionTensor3D<double> TensorType;
typedef itk::Image<TensorType,nDims> InputImageType;
typedef itk::Image<int,nDims> SegImageType;
typedef itk::Image<double,nDims> FAImageType;
typedef itk::Image<int,nDims> OutputImageType;
typedef InputImageType::IndexType IndexType; // all 4 image types have the same index type
typedef InputImageType::SizeType SizeType;   // all 4 image types have the same size type

// find out whether a voxel is greater than zero and less than size
bool inRange(IndexType vox, SizeType size) {
  for (int d = 0; d < nDims; d++)
    if (vox[d] < 0 || vox[d] >= size[d])
      return false;
  return true;
}

// get size of the image
template<typename Img>
SizeType getSize(typename Img::Pointer img) {
  return img->GetLargestPossibleRegion().GetSize();
}

// read an image file with a given filename
template<typename Img>
typename Img::Pointer readImgFile(char* fileName) {
  typename itk::ImageFileReader<Img>::Pointer reader = itk::ImageFileReader<Img>::New();
  reader->SetFileName(fileName);
  reader->Update();
  return reader->GetOutput();
}

// run fractional anisotropy filter on an image
FAImageType::Pointer runFAFilter(InputImageType::Pointer input) {
  typedef itk::TensorFractionalAnisotropyImageFilter<InputImageType, FAImageType> FAFilterType;
  FAFilterType::Pointer faFilter = FAFilterType::New();
  faFilter->SetInput(input);
  faFilter->Update();
  return faFilter->GetOutput();
}

// get principal eigenvector of a tensor; then multiply it by delta
SizeType getPrincipalEigenVec(TensorType pixel, double delta) {
  TensorType::EigenValuesArrayType eigenVals;
  TensorType::EigenVectorsMatrixType eigenVecs;
  pixel.ComputeEigenAnalysis(eigenVals,eigenVecs);

  int maxIndex = -1;
  double maxVal = -1;
  for (int i = 0; i < nDims; i++) {
    double thisVal = abs(eigenVals[i]);
    if (thisVal > maxVal) {
      maxVal = thisVal;
      maxIndex = i;
    }
  }

  SizeType returnVal;
  for (int d = 0; d < nDims; d++) {
    returnVal[d] = round(eigenVecs(d,maxIndex)*delta);
  }
  return returnVal;
}

// make a blank image with the same size as img1
template<typename Img1, typename Img2>
typename Img2::Pointer makeBlankImage(typename Img1::Pointer img1) {
  typename Img2::Pointer img2 = Img2::New();
  img2->SetRegions(getSize<Img1>(img1));
  img2->SetOrigin(img1->GetOrigin());
  img2->SetDirection(img1->GetDirection());
  img2->SetSpacing(img1->GetSpacing());
  img2->Allocate();
  return img2;
}

// make an iterator that goes through the entire image img1
template<typename Img>
typename itk::ImageRegionIterator<Img> makeFullImgIterator(typename Img::Pointer img) {
  IndexType index;

  typename Img::RegionType fullRegion;
  fullRegion.SetSize(getSize<Img>(img));
  fullRegion.SetIndex(index);

  return itk::ImageRegionIterator<Img>(img, fullRegion);
}

class TractQueueEntry {
public:
  IndexType voxel;
  int iterNumber;
  int color;

  TractQueueEntry(IndexType voxel, int iterNumber, int color) : voxel(voxel), iterNumber(iterNumber), color(color) {}
};

class TractographyCommand : public vtkCommand {
public:

  static TractographyCommand* New() {
    return new TractographyCommand();
  }

  void setParams(double _delta, double _minFA, int _maxIter, InputImageType::Pointer _input, FAImageType::Pointer _fa, vtkSmartPointer<vtkRenderer> _renderer2, double _centerDiff[3]) {
    delta = _delta;
    minFA = _minFA;
    maxIter = _maxIter;
    input = _input;
    fa = _fa;
    size = getSize<InputImageType>(input);
    alreadyTouched = makeBlankImage<InputImageType, SegImageType>(input);
    renderer2 = _renderer2;
    for (int i = 0; i < 3; i++) centerDiff[i] = _centerDiff[i];
  }

  void addPoint(IndexType point) {
    alreadyTouched->SetPixel(point, 1);
    voxelsQueue.push({ point, 1, maxColorReached });
    maxColorReached++;
  }

  void addSeg(SegImageType::Pointer seg) {
    itk::ImageRegionIterator<SegImageType> iterator = makeFullImgIterator<SegImageType>(seg);
    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
      if (iterator.Get() != 0) {
        alreadyTouched->SetPixel(iterator.GetIndex(), 1);
        voxelsQueue.push({ iterator.GetIndex(), 1, maxColorReached });
      }
    maxColorReached++;
  }

  virtual void Execute (vtkObject *caller, unsigned long eventId, void *callData) {
    if (eventId == vtkCommand::RightButtonPressEvent) {
      vtkSmartPointer < vtkRenderWindowInteractor > interactor = dynamic_cast < vtkRenderWindowInteractor * > ( caller ) ;
      vtkSmartPointer<vtkRenderer> renderer = interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer();

      int clickLocation[2];
      interactor->GetEventPosition(clickLocation);

      vtkSmartPointer<vtkPointPicker> picker = dynamic_cast<vtkPointPicker*>(interactor->GetPicker());
      int pickWorked = picker->Pick(clickLocation[0], clickLocation[1], 0, renderer); // 0 bc focal point???
      double pickedPos[3];
      picker->GetPickPosition(pickedPos);

      for (int i = 0; i < 3; i++) pickedPos[i] -= centerDiff[i];

      itk::Point<double, 3> tmp;
      for (int i = 0; i < 3; i++) tmp[i] = pickedPos[i];
      itk::ContinuousIndex<double, 3> index = fa->TransformPhysicalPointToContinuousIndex<double>(tmp);

      addPoint({(long)round(index[0]),(long)round(index[1]),(long)round(index[1])});
    } else if (eventId == vtkCommand::TimerEvent) {
      vtkSmartPointer < vtkRenderWindowInteractor > interactor = dynamic_cast < vtkRenderWindowInteractor * > ( caller ) ;
      vtkSmartPointer<vtkRenderer> renderer = interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer();
      queueExpandTick();
      interactor->Render();
    }
  }

private:

  double delta;
  double minFA;
  int maxIter;
  InputImageType::Pointer input;
  FAImageType::Pointer fa;
  SegImageType::Pointer alreadyTouched;
  SizeType size;
  vtkSmartPointer<vtkRenderer> renderer2;
  std::queue<TractQueueEntry> voxelsQueue;
  int maxColorReached = 0;
  // std::vec<vtkSmartPointer<vtkPolyDataMapper>> mappers;
  double centerDiff[3];

  void lookupColor(int color, double colorDoubles[3]) {
    color %= 6;
         if (color == 0) { colorDoubles[0] = 1; colorDoubles[1] = 0; colorDoubles[2] = 0; }
    else if (color == 1) { colorDoubles[0] = 0; colorDoubles[1] = 1; colorDoubles[2] = 0; }
    else if (color == 2) { colorDoubles[0] = 0; colorDoubles[1] = 0; colorDoubles[2] = 1; }
    else if (color == 3) { colorDoubles[0] = 1; colorDoubles[1] = 1; colorDoubles[2] = 0; }
    else if (color == 4) { colorDoubles[0] = 0; colorDoubles[1] = 1; colorDoubles[2] = 1; }
    else if (color == 5) { colorDoubles[0] = 1; colorDoubles[1] = 0; colorDoubles[2] = 1; }
  }

  void addLineActor(double point1[3], double point2[3], int color) {
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(point1);
    line->SetPoint2(point2);
    line->SetResolution(1);
    line->Update();

    // vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
    // sphere->SetCenter(point1);
    // sphere->SetRadius(.1);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(line->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    double colorDoubles[3];
    lookupColor(color, colorDoubles);
    actor->GetProperty()->SetColor(colorDoubles);
    actor->GetProperty()->SetLineWidth(1);
    actor->SetMapper(mapper);

    renderer2->AddActor(actor);
  }

  void queueExpandTick() {
    std::queue<TractQueueEntry> newVoxelsQueue;

    while (!voxelsQueue.empty()) {
      TractQueueEntry thisEntry = voxelsQueue.front();
      voxelsQueue.pop();

      if (fa->GetPixel(thisEntry.voxel) < minFA || thisEntry.iterNumber > maxIter)
        continue;

      SizeType eigenVecScaled = getPrincipalEigenVec(input->GetPixel(thisEntry.voxel), delta);
      IndexType addedVoxel  = thisEntry.voxel + eigenVecScaled;
      IndexType subbedVoxel = thisEntry.voxel - eigenVecScaled;

      if (inRange(addedVoxel, size) && alreadyTouched->GetPixel(addedVoxel) == 0) {
        alreadyTouched->SetPixel(addedVoxel, 1);
        double point1[3];
        double point2[3];
        for (int i = 0; i < 3; i++) { point1[i] = thisEntry.voxel[i]; point2[i] = addedVoxel[i]; }
        addLineActor(point1, point2, thisEntry.color);
        newVoxelsQueue.push({ addedVoxel, thisEntry.iterNumber+1, thisEntry.color });
      }

      // if (inRange(subbedVoxel, size) && alreadyTouched->GetPixel(subbedVoxel) == 0) {
      //   alreadyTouched->SetPixel(subbedVoxel, 1);
      //   double point1[3];
      //   double point2[3];
      //   for (int i = 0; i < 3; i++) { point1[i] = thisEntry.voxel[i]; point2[i] = subbedVoxel[i]; }
      //   addLineActor(point1, point2, thisEntry.color);
      //   newVoxelsQueue.push({ subbedVoxel, thisEntry.iterNumber+1, thisEntry.color });
      // }
    }

    voxelsQueue = newVoxelsQueue;
  }
};

// TODO maybe remove this
class NoRightClickStyle : public vtkInteractorStyleImage {
public:
  NoRightClickStyle() { vtkInteractorStyleImage::SetInteractionModeToImage3D() ; }
  static NoRightClickStyle* New() { return new NoRightClickStyle(); }
  virtual void RightButtonPressEvent() {}
};

int main ( int argc, char * argv[] )
{
  if (argc < 4 || argc > 8) {
    std::cout << "Wrong number of args; expected: " << argv[0] << " inputImage inputSegmentation outputImage (delta = 3) (minFA = 0.2) (maxIter = 300) (outputSegmentation = \"\")" << std::endl;
    return 1;
  }

  InputImageType::Pointer input = readImgFile<InputImageType>(argv[1]);
  SegImageType  ::Pointer seg   = readImgFile<SegImageType>(argv[2]);

  double delta = (argc > 4) ? strtof(argv[4],NULL) : 3;
  double minFA = (argc > 5) ? strtof(argv[5],NULL) : .01;
  int maxIter  = (argc > 6) ? strtod(argv[6],NULL) : 10; // 300;

  if (delta == 0) {
    std::cout << "Bad value for delta" << std::endl;
    return 1;
  }
  if (minFA == 0) {
    std::cout << "Bad value for minFA" << std::endl;
    return 1;
  }
  if (maxIter == 0) {
    std::cout << "Bad value for maxIter" << std::endl;
    return 1;
  }

  FAImageType::Pointer fa = runFAFilter(input);

  typedef itk::MultiplyImageFilter<FAImageType, FAImageType, FAImageType> MultFilterType;
  MultFilterType::Pointer multFilter = MultFilterType::New();
  multFilter->SetInput(fa);
  multFilter->SetConstant(200);
  multFilter->Update();

  typedef itk::ImageToVTKImageFilter < FAImageType > ITKToVTKFilterType ;
  ITKToVTKFilterType::Pointer itkToVTKfilter = ITKToVTKFilterType::New() ;
  itkToVTKfilter->SetInput ( multFilter->GetOutput() ) ;
  itkToVTKfilter->Update() ;

  vtkSmartPointer < vtkImageSliceMapper > imageMapper = vtkSmartPointer < vtkImageSliceMapper > ::New() ;
  imageMapper->SetInputData ( itkToVTKfilter->GetOutput() ) ;
  imageMapper->SetOrientationToX () ;
  imageMapper->SetSliceNumber ( 55 ) ;
  imageMapper->SliceAtFocalPointOn () ;
  imageMapper->SliceFacesCameraOn () ;

  vtkSmartPointer < vtkImageActor > imageActor = vtkSmartPointer < vtkImageActor > ::New() ;
  imageActor->SetMapper ( imageMapper ) ;

  vtkSmartPointer < vtkRenderer > renderer = vtkSmartPointer < vtkRenderer >::New() ;
  renderer->AddActor ( imageActor ) ;

    
  vtkSmartPointer < vtkCamera > camera = renderer->GetActiveCamera() ;

  double position[3],  imageCenter[3] ;
  itkToVTKfilter->GetOutput()->GetCenter ( imageCenter ) ;
  position[0] = imageCenter[0] ;
  position[1] = imageCenter[1] ;
  position[2] = -160 ;
  double spacing[3] ;
  int imageDims[3] ;
  itkToVTKfilter->GetOutput()->GetSpacing ( spacing ) ;
  itkToVTKfilter->GetOutput()->GetDimensions ( imageDims ) ;
  double imagePhysicalSize[3] ;
  for ( unsigned int d = 0 ; d < 3 ; d++ )
    {
      imagePhysicalSize[d] = spacing[d] * imageDims[d] ;
    }

  camera->ParallelProjectionOn () ; 
  camera->SetFocalPoint ( imageCenter ) ;
  camera->SetPosition ( position ) ;
  camera->SetParallelScale ( imageDims[2] ) ;

  SizeType size = fa->GetLargestPossibleRegion().GetSize();
  double centerCoord[3];
  for (int i = 0; i < 3; i++) centerCoord[i] = ((double) size[i])/2;
  itk::Point<double,3> itkCenter = fa->TransformIndexToPhysicalPoint<double>({(long)centerCoord[0],(long)centerCoord[1],(long)centerCoord[2]});

  double centerDiff[3];
  for (int i = 0; i < 3; i++) centerDiff[i] = imageCenter[i]-itkCenter[i];


  vtkSmartPointer < vtkRenderer > renderer2 = vtkSmartPointer < vtkRenderer >::New() ;
  vtkSmartPointer<vtkCamera> camera2 = renderer2->GetActiveCamera();
  camera2->SetFocalPoint(72,72,42);


  renderer->SetViewport(0,0,.5,1);
  renderer2->SetViewport(.5,0,1,1);


  vtkSmartPointer < vtkRenderWindow > window = vtkSmartPointer < vtkRenderWindow >::New() ;
  window->AddRenderer ( renderer ) ;
  window->AddRenderer ( renderer2 ) ;
  window->SetSize ( 1000, 500 ) ;

  vtkSmartPointer < vtkRenderWindowInteractor > interactor = vtkSmartPointer < vtkRenderWindowInteractor >::New() ;
  interactor->SetRenderWindow ( window ) ;

  vtkSmartPointer < NoRightClickStyle > style = vtkSmartPointer < NoRightClickStyle >::New() ;
  // style->SetInteractionModeToImageSlicing() ;
  // interactor->SetInteractorStyle ( style ) ;
  interactor->Initialize();
  interactor->CreateRepeatingTimer(1000);

  vtkSmartPointer<vtkPointPicker> picker = vtkSmartPointer<vtkPointPicker>::New();
  interactor->SetPicker(picker);

  vtkSmartPointer<TractographyCommand> myCallback = vtkSmartPointer<TractographyCommand>::New();
  myCallback->setParams(delta, minFA, maxIter, input, fa, renderer2, centerDiff);
  myCallback->addPoint({72, 72, 42});
  myCallback->addSeg(seg);
  interactor->AddObserver(vtkCommand::TimerEvent, myCallback, 0);
  interactor->AddObserver(vtkCommand::RightButtonPressEvent, myCallback, 0);

  interactor->Start() ;

  return 0 ;
}

