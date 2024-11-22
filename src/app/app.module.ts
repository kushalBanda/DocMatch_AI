import { CUSTOM_ELEMENTS_SCHEMA, NgModule } from '@angular/core';
import { ProgressBarModule } from 'primeng/progressbar';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { BrowserModule } from '@angular/platform-browser';
import { MultiSelectModule } from 'primeng/multiselect';
import { ConfirmationService, MessageService } from 'primeng/api';
import { ConfirmDialogModule } from 'primeng/confirmdialog';
import { AppComponent } from './app.component';
import { DocMatchComponent } from './doc-match/doc-match.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { TooltipModule } from 'primeng/tooltip';
import { TabViewModule } from 'primeng/tabview';
import { OverlayPanelModule } from 'primeng/overlaypanel';
import { TableModule } from 'primeng/table';
import { NgxSpinnerModule } from 'ngx-spinner';
import { ButtonModule } from 'primeng/button';
import { SidebarModule } from 'primeng/sidebar';
import { SkeletonModule } from 'primeng/skeleton';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import { PdfViewerComponent } from './PdfViewer/pdf-viewer/pdf-viewer.component';
import { MatDialogModule } from '@angular/material/dialog';
import { PdfViewerPopupComponent } from './PdfViewer/pdf-viewer-popup/pdf-viewer-popup.component';
import {MatIconModule} from '@angular/material/icon';
import { NgApexchartsModule } from "ng-apexcharts";

@NgModule({
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  declarations: [
    AppComponent,
    DocMatchComponent,
    PdfViewerComponent,
    PdfViewerPopupComponent
  ],
  imports: [
    SkeletonModule,
    NgxSpinnerModule,
    ButtonModule,
    SidebarModule,
    ConfirmDialogModule,
    ProgressSpinnerModule,
    OverlayPanelModule,
    TabViewModule,
    HttpClientModule,
    ProgressBarModule,
    TooltipModule,
    FormsModule,
    BrowserAnimationsModule,
    BrowserModule,
    CommonModule,
    TableModule,
    MultiSelectModule,
    MatProgressSpinnerModule,
    MatDialogModule,
    MatIconModule,
    NgApexchartsModule

  ],
  providers: [ConfirmationService, MessageService],
  bootstrap: [AppComponent]
})
export class AppModule { }
