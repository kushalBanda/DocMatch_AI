import { ChangeDetectorRef, Component, ElementRef, EventEmitter, Inject, Input, OnInit, Output, ViewChild, ViewEncapsulation } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import * as PDFJS from 'pdfjs-dist';
import { MatMenu, MatMenuTrigger } from '@angular/material/menu';
import { MatIcon } from '@angular/material/icon';
import {
  ApexAxisChartSeries,
  ApexTitleSubtitle,
  ApexDataLabels,
  ApexChart,
  ChartComponent,
  ApexPlotOptions,
  ApexXAxis,
  ApexYAxis
} from "ng-apexcharts";



@Component({
  selector: 'app-pdf-viewer-popup',
  templateUrl: './pdf-viewer-popup.component.html',
  styleUrls: ['./pdf-viewer-popup.component.css'],

})

export class PdfViewerPopupComponent implements OnInit {
 @Input() docsdata !:any;
  @ViewChild("chart") chart!: ChartComponent;
  heapmap = '../../../assets/heatmap/heatmap.png'
  heatmapSrc: string = '../../../assets/Heatmaps/heatmap.png';
  
  constructor() {
   
  }

  getDocumentEntries() {
    return Object.entries(this.docsdata);
  }
  async ngOnInit(): Promise<void> {
    this.reloadHeatmap();
  }

  reloadHeatmap(): void {
    // Append a timestamp to the src URL to force the browser to load the new image
    this.heatmapSrc = '../../../assets/Heatmaps/heatmap.png?v=' + new Date().getTime();
  }

}



