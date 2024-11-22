import { Component, Inject, OnInit,AfterViewInit, HostListener, ViewChild, ElementRef, Input  } from '@angular/core';
import { inject } from '@angular/core/testing';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { PdfViewerPopupComponent } from '../pdf-viewer-popup/pdf-viewer-popup.component';
import * as pdfjsLib  from 'pdfjs-dist';

import 'pdfjs-dist/build/pdf.worker.entry';
import { HttpClient } from '@angular/common/http';
import { DataService } from 'src/app/data.service';
@Component({
  selector: 'app-pdf-viewer',
  templateUrl: './pdf-viewer.component.html',
  styleUrls: ['./pdf-viewer.component.css']
})
export class PdfViewerComponent implements OnInit {
  pdfSrc1 = this.sanitizeUrl("../../../assets/highlighted_pdfs/SOP.EG.002 E0_637520196028800408_highlighted.pdf");  // Path to your first PDF
  pdfSrc2 = this.sanitizeUrl("../../../assets/highlighted_pdfs/SOP.EG.002 E1_637553751272743741_highlighted.pdf");
  //pdfSrc1:string = '../../../assets/SOP-SE-026  Preventive Maintenance of Fluid Bed Dryer_638295235050509182.pdf';  
  //pdfSrc2:string = '../../../assets/SOP.SE.026.SOP on Preventive Maintenance of Fluid Bed Dryer_637488108045519011.pdf';
  pdfSrc:string = '../../../assets/SOP-SE-026  Preventive Maintenance of Fluid Bed Dryer_638295235050509182.pdf';  

  @ViewChild('pdfContainer', { static: true }) pdfContainer!: ElementRef;
  //@ViewChild('pdfCanvas1', { static: true }) pdfCanvas1!: ElementRef<HTMLCanvasElement>;
  //@ViewChild('pdfCanvas2', { static: true }) pdfCanvas2!: ElementRef<HTMLCanvasElement>;
  //@ViewChild('highlightCanvas1', { static: true }) highlightCanvas1!: ElementRef<HTMLCanvasElement>;
  //@ViewChild('highlightCanvas2', { static: true }) highlightCanvas2!: ElementRef<HTMLCanvasElement>;

  // highlightCoordinates = [
  //   { x: 100, y: 150, width: 200, height: 50 },
  //   // Add more coordinates as needed
  // ];
  // isDrawing = false;
  // startX = 0;
  // startY = 0;
  
  pdfDoc1: any;
  pdfDoc2: any;
  pageNum1 = 1;
  pageNum2 = 1;
  highlights1:any ;
  highlights2:any ;
  basePath: string ='../../../assets/highlighted_pdfs';

  constructor(
    @Inject(MAT_DIALOG_DATA) public data: any,
    private elRef: ElementRef,
    private dialogRef: MatDialogRef<PdfViewerComponent>,
    private sanitizer: DomSanitizer, 
    private pdfService:DataService
  ) { 
    const pdfWorkerSrc = 'pdf.worker.min.js';
    pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;
  }

  

  async ngOnInit(): Promise<void> {
    // const pdfPath = 'assets/your_pdf_file.pdf'; // Adjust this path to match your local PDF file
    // const department = this.data.department;

    let pathParts1 = this.data.response.highlighted_pdf1_path.split('\\');
    let fileName1 = pathParts1[pathParts1.length - 1];
    let pathParts2 = this.data.response.highlighted_pdf2_path.split('\\');
    let fileName2 = pathParts2[pathParts2.length - 1];

    this.pdfSrc1 =this.sanitizeUrl(this.basePath + '/'+ fileName1);
    this.pdfSrc2 = this.sanitizeUrl(this.basePath + '/'+ fileName2);//this.sanitizeUrl(encodeURI(`${this.data.response.highlighted_pdf2_path}`));
    // if(this.data.companypair.length==2){
    //   const companyPair1 = this.data.companypair[0];
    //   const companyPair2 = this.data.companypair[1];
    //   this.pdfSrc1 = `${basePath}/${companyPair1}/${department}/${encodeURIComponent(this.data.pdfsrc1)}`;
    //   this.pdfSrc2 =`${basePath}/${companyPair2}/${department}/${encodeURIComponent(this.data.pdfsrc2)}`;

    // }
    // else if(this.data.companypair.length==1){
    //   const companyPair = this.data.companypair[0];
    //   this.pdfSrc1 =`${basePath}/${companyPair}/${department}/${encodeURIComponent(this.data.pdfsrc1)}`;
    //   this.pdfSrc2 =`${basePath}/${companyPair}/${department}/${encodeURIComponent(this.data.pdfsrc2)}`;  
        
    // }
    // this.highlights1 = this.data.response.pdf1;
    // this.highlights2= this.data.response.pdf2;
    // this.loadPdf(this.pdfSrc1, 1);
    // this.loadPdf(this.pdfSrc2, 2);
 
  }
  Close(): void {
    this.dialogRef.close(true);
  }

 

  // loadPdf(url: string, viewerNumber: number) {
  //   const encodedUrl = decodeURI(url);
  //   pdfjsLib.getDocument(encodedUrl).promise.then((pdfDoc) => {
  //     if (viewerNumber === 1) {
  //       this.pdfDoc1 = pdfDoc;
  //       this.renderPage(this.pageNum1, 1);
  //     } else {
  //       this.pdfDoc2 = pdfDoc;
  //       this.renderPage(this.pageNum2, 2);
  //     }
  //   });
  // }
  // renderPage(pageNum: number, viewerNumber: number) {
  //   let canvas = viewerNumber === 1 ? this.pdfCanvas1.nativeElement : this.pdfCanvas2.nativeElement;
  //   let highlightCanvas = viewerNumber === 1 ? this.highlightCanvas1.nativeElement : this.highlightCanvas2.nativeElement;
  //   let pdfDoc = viewerNumber === 1 ? this.pdfDoc1 : this.pdfDoc2;

  //   pdfDoc.getPage(pageNum).then((page:any) => {
  //     let viewport = page.getViewport({ scale: 1.0 });
  //     let context = canvas.getContext('2d');
  //     let highlightContext = highlightCanvas.getContext('2d');

  //     if (context && highlightContext) {
  //       canvas.height = viewport.height;
  //       canvas.width = viewport.width;
  //       highlightCanvas.height = viewport.height;
  //       highlightCanvas.width = viewport.width;

  //       let renderContext = {
  //         canvasContext: context,
  //         viewport: viewport
  //       };
  //       page.render(renderContext).promise.then(() => {
  //         this.renderHighlights(viewerNumber);
  //       });
  //     }
  //   });
  // }

  // renderHighlights(viewerNumber: number) {
  //   let canvas = viewerNumber === 1 ? this.highlightCanvas1.nativeElement : this.highlightCanvas2.nativeElement;
  //   let context = canvas.getContext('2d');
  //   if (context) {
  //     let highlights = viewerNumber === 1 ? this.highlights1[this.pageNum1] : this.highlights2[this.pageNum2];
  //     //let highlightsred = viewerNumber === 1 ? this.highlights1.red[this.pageNum1] : this.highlights2.red[this.pageNum2];

  //     context.clearRect(0, 0, canvas.width, canvas.height);
  //     if (highlights) {
  //       context.fillStyle = 'rgba(59, 166, 204, 0.29)';
  //       highlights.forEach((rect: any) => {
  //         context?.fillRect(rect.x-35, rect.y-10, rect.width-20, rect.height);
  //       });
  //     }
  //     // if (highlightsred) {
  //     //   context.fillStyle = 'rgba(233, 63, 11, 0.57)';
  //     //   highlightsred.forEach((rect: any) => {
  //     //     context?.fillRect(rect.x-35, rect.y-10, rect.width-20, rect.height);
  //     //   });
  //     // }
  //   }
  // }

  // nextPage(pdf:string) {
  //   if (this.pageNum1 < this.pdfDoc1.numPages && pdf =="pdf1") {
  //     this.pageNum1++;
  //     this.renderPage(this.pageNum1, 1);
  //   }
  //   if (this.pageNum2 < this.pdfDoc2.numPages && pdf =="pdf2") {
  //     this.pageNum2++;
  //     this.renderPage(this.pageNum2, 2);
  //   }
  // }

  // previousPage(pdf:string) {
  //   if (this.pageNum1 > 1&& pdf =="pdf1") {
  //     this.pageNum1--;
  //     this.renderPage(this.pageNum1, 1);
  //   }
  //   if (this.pageNum2 > 1&& pdf =="pdf2") {
  //     this.pageNum2--;
  //     this.renderPage(this.pageNum2, 2);
  //   }
  // }

  // @HostListener('mousedown', ['$event'])
  // onMouseDown(event: MouseEvent) {
  //   this.isDrawing = true;
  //   this.startX = event.offsetX;
  //   this.startY = event.offsetY;
  // }

  // @HostListener('mouseup', ['$event'])
  // onMouseUp(event: MouseEvent) {
  //   if (!this.isDrawing) return;
  //   this.isDrawing = false;

  //   let endX = event.offsetX;
  //   let endY = event.offsetY;
  //   let rect = {
  //     x: Math.min(this.startX, endX),
  //     y: Math.min(this.startY, endY),
  //     width: Math.abs(this.startX - endX),
  //     height: Math.abs(this.startY - endY)
  //   };

  //   if (event.target === this.highlightCanvas1.nativeElement) {
  //     if (!this.highlights1[this.pageNum1]) this.highlights1[this.pageNum1] = [];
  //     this.highlights1[this.pageNum1].push(rect);
  //     this.renderHighlights(1);
  //   } else if (event.target === this.highlightCanvas2.nativeElement) {
  //     if (!this.highlights2[this.pageNum2]) this.highlights2[this.pageNum2] = [];
  //     this.highlights2[this.pageNum2].push(rect);
  //     this.renderHighlights(2);
  //   }
  // }
 
  
  

  // highlightText(context: CanvasRenderingContext2D, viewport: any): void {
  //   context.fillStyle = 'rgba(255, 255, 0, 0.5)'; // Semi-transparent yellow
  //   this.highlightCoordinates.forEach(coord => {
  //     const [x, y, width, height] = this.normalizeCoordinates(coord, viewport);
  //     context.fillRect(x, y, width, height);
  //   });
  // }

  // normalizeCoordinates(coord: any, viewport: any): [number, number, number, number] {
  //   // Normalize coordinates based on viewport transformation
  //   const transform = viewport.transform;
  //   const x = coord.x * transform[0] + transform[4];
  //   const y = coord.y * transform[3] + transform[5];
  //   const width = coord.width * transform[0];
  //   const height = coord.height * transform[3];
  //   return [x, y, width, height];
  // }
  sanitizeUrl(url: string): SafeResourceUrl {
    return this.sanitizer.bypassSecurityTrustResourceUrl(url);
  }

}
