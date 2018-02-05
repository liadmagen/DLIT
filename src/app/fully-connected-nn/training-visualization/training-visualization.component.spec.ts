import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainingVisualizationComponent } from './training-visualization.component';

describe('TrainingVisualizationComponent', () => {
  let component: TrainingVisualizationComponent;
  let fixture: ComponentFixture<TrainingVisualizationComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TrainingVisualizationComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(TrainingVisualizationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
